class MemoryMappedFile {
public:
  MemoryMappedFile(const std::string &path, size_t offset_bytes,
                   size_t size_bytes)
      : path_(path), offset_bytes_(offset_bytes), size_bytes_(size_bytes) {
#if defined(__linux__)
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      return;
    }

    struct stat st{};
    if (fstat(fd_, &st) != 0 || st.st_size < 0 ||
        offset_bytes_ > std::numeric_limits<size_t>::max() - size_bytes_ ||
        static_cast<uintmax_t>(offset_bytes_ + size_bytes_) >
            static_cast<uintmax_t>(st.st_size)) {
      close(fd_);
      fd_ = -1;
      return;
    }

    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (page_size <= 0) {
      close(fd_);
      fd_ = -1;
      return;
    }

    const size_t page_mask = static_cast<size_t>(page_size - 1);
    const size_t aligned_offset = offset_bytes_ & ~page_mask;
    map_shift_bytes_ = offset_bytes_ - aligned_offset;
    if (size_bytes_ > std::numeric_limits<size_t>::max() - map_shift_bytes_) {
      close(fd_);
      fd_ = -1;
      map_shift_bytes_ = 0;
      return;
    }
    mapped_size_bytes_ = size_bytes_ + map_shift_bytes_;

    mapped_data_ = mmap(nullptr, mapped_size_bytes_, PROT_READ, MAP_PRIVATE,
                        fd_, static_cast<off_t>(aligned_offset));
    if (mapped_data_ == MAP_FAILED) {
      mapped_data_ = nullptr;
      mapped_size_bytes_ = 0;
      map_shift_bytes_ = 0;
      close(fd_);
      fd_ = -1;
    }
#endif
  }

  ~MemoryMappedFile() {
#if defined(__linux__)
    if (mapped_data_ != nullptr) {
      munmap(mapped_data_, mapped_size_bytes_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
#endif
  }

  [[nodiscard]] std::vector<double> readChunk(size_t start_idx, size_t count,
                                              size_t point_dim) const {
    if (point_dim != 0 &&
        count > std::numeric_limits<size_t>::max() / point_dim) {
      throw std::overflow_error("tile chunk shape overflows size_t");
    }
    std::vector<double> data(count * point_dim);
    if (data.empty()) {
      return data;
    }

    if (point_dim != 0 &&
        start_idx > std::numeric_limits<size_t>::max() / point_dim) {
      throw std::overflow_error("tile chunk offset overflows size_t");
    }
    const size_t offset_values = start_idx * point_dim;
    if (offset_values > std::numeric_limits<size_t>::max() / sizeof(double) ||
        data.size() > std::numeric_limits<size_t>::max() / sizeof(double)) {
      throw std::overflow_error("tile chunk byte range overflows size_t");
    }
    const size_t chunk_offset_bytes = offset_values * sizeof(double);
    const size_t chunk_size_bytes = data.size() * sizeof(double);

#if defined(__linux__)
    if (mapped_data_ != nullptr &&
        chunk_offset_bytes + chunk_size_bytes <= size_bytes_) {
      const auto *src = static_cast<const char *>(mapped_data_) +
                        map_shift_bytes_ + chunk_offset_bytes;
      std::memcpy(data.data(), src, chunk_size_bytes);
      return data;
    }
#endif

    std::ifstream file(path_, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open input file: " + path_);
    }

    const auto seek_to =
        static_cast<std::streamoff>(offset_bytes_ + chunk_offset_bytes);
    file.seekg(seek_to, std::ios::beg);
    if (!file) {
      throw std::runtime_error("Failed to seek in input file: " + path_);
    }

    file.read(reinterpret_cast<char *>(data.data()),
              static_cast<std::streamsize>(chunk_size_bytes));
    if (!file) {
      throw std::runtime_error(
          "Failed to read expected chunk from input file: " + path_);
    }

    return data;
  }

private:
  std::string path_;
  size_t offset_bytes_ = 0;
  size_t size_bytes_ = 0;

#if defined(__linux__)
  int fd_ = -1;
  void *mapped_data_ = nullptr;
  size_t mapped_size_bytes_ = 0;
  size_t map_shift_bytes_ = 0;
#endif
};

class TilePartitioner {
public:
  struct Tile {
    size_t start_idx = 0;      // Start index including left overlap.
    size_t local_count = 0;    // Number of points in this tile window.
    size_t overlap_before = 0; // Number of points shared with previous tile.
    size_t overlap_after = 0;  // Number of points shared with next tile.
    std::vector<double> bounds;
  };

  [[nodiscard]] static std::vector<Tile> partitionLinear(size_t num_points,
                                                         size_t max_tile_size,
                                                         double overlap_ratio) {
    std::vector<Tile> tiles;
    if (num_points == 0 || max_tile_size == 0) {
      return tiles;
    }

    if (!std::isfinite(overlap_ratio) || overlap_ratio < 0.0 ||
        overlap_ratio > 0.5) {
      throw std::invalid_argument(
          "tile overlap_ratio must be finite and in [0, 0.5]");
    }
    const size_t overlap_points =
        std::min(max_tile_size > 1 ? max_tile_size - 1 : static_cast<size_t>(0),
                 static_cast<size_t>(static_cast<double>(max_tile_size) *
                                     overlap_ratio));

    for (size_t core_start = 0; core_start < num_points;) {
      const size_t core_count =
          std::min(max_tile_size, num_points - core_start);
      const size_t core_end = core_start + core_count;
      const size_t read_start =
          (core_start > overlap_points) ? (core_start - overlap_points) : 0;
      const size_t read_end =
          core_end + std::min(overlap_points, num_points - core_end);

      Tile tile;
      tile.start_idx = read_start;
      tile.local_count = read_end - read_start;
      tile.overlap_before = core_start - read_start;
      tile.overlap_after = read_end - core_end;
      tiles.push_back(std::move(tile));

      core_start = core_end;
    }

    return tiles;
  }

  [[nodiscard]] static std::vector<Tile>
  partitionPoints(const std::vector<double> &points, size_t point_dim,
                  size_t num_points, size_t max_tile_size,
                  double overlap_ratio = 0.1) {
    auto tiles = partitionLinear(num_points, max_tile_size, overlap_ratio);
    for (auto &tile : tiles) {
      tile.bounds =
          computeBounds(points, point_dim, tile.start_idx, tile.local_count);
    }
    return tiles;
  }

private:
  [[nodiscard]] static std::vector<double>
  computeBounds(const std::vector<double> &points, size_t point_dim,
                size_t start, size_t count) {
    if (point_dim > std::numeric_limits<size_t>::max() / 2) {
      throw std::overflow_error("tile bounds dimension overflows size_t");
    }
    std::vector<double> bounds(point_dim * 2, 0.0);
    if (point_dim == 0 || count == 0 || points.empty()) {
      return bounds;
    }

    for (size_t d = 0; d < point_dim; ++d) {
      bounds[d * 2] = std::numeric_limits<double>::infinity();
      bounds[d * 2 + 1] = -std::numeric_limits<double>::infinity();
    }

    const size_t num_total_points = points.size() / point_dim;
    if (start >= num_total_points) {
      return bounds;
    }
    const size_t end = start + std::min(count, num_total_points - start);
    for (size_t i = start; i < end; ++i) {
      for (size_t d = 0; d < point_dim; ++d) {
        const double value = points[i * point_dim + d];
        bounds[d * 2] = std::min(bounds[d * 2], value);
        bounds[d * 2 + 1] = std::max(bounds[d * 2 + 1], value);
      }
    }

    return bounds;
  }
};

class ThreadPool {
public:
  explicit ThreadPool(size_t num_threads) {
    const size_t threads = std::max<size_t>(1, num_threads);
    workers_.reserve(threads);
    for (size_t i = 0; i < threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
              return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  template <typename F, typename... Args>
  auto enqueue(F &&f, Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>> {
    using ReturnType = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<ReturnType> result = task->get_future();

    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      tasks_.emplace([task] { (*task)(); });
    }

    condition_.notify_one();
    return result;
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    for (auto &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_ = false;
};

class AsyncTileProcessor {
public:
  explicit AsyncTileProcessor(size_t num_threads) : pool_(num_threads) {}

  [[nodiscard]] std::future<std::vector<Pair>>
  submitTile(const std::vector<double> &tile_points, size_t point_dim,
             const FloodComplexConfig &config) {
    return pool_.enqueue([tile_points, point_dim, config]() {
      const size_t num_points =
          point_dim == 0 ? 0 : (tile_points.size() / point_dim);
      auto result =
          computeFloodComplex(tile_points, point_dim, num_points, config);
      return result.pairs;
    });
  }

private:
  ThreadPool pool_;
};

class TileResultMerger {
public:
  [[nodiscard]] static std::vector<Pair>
  mergeResults(const std::vector<std::vector<Pair>> &tile_results,
               const std::vector<TilePartitioner::Tile> &tiles,
               double merge_tolerance = kDefaultTileMergeTolerance) {
    if (!tiles.empty() && tiles.size() != tile_results.size()) {
      throw std::invalid_argument("tile metadata must match tile result count");
    }
    std::vector<Pair> merged;
    size_t reserve_pairs = 0;
    for (const auto &tile_pairs : tile_results) {
      if (reserve_pairs >
          std::numeric_limits<size_t>::max() - tile_pairs.size()) {
        reserve_pairs = std::numeric_limits<size_t>::max();
        break;
      }
      reserve_pairs += tile_pairs.size();
    }
    if (reserve_pairs != std::numeric_limits<size_t>::max()) {
      merged.reserve(reserve_pairs);
    }

    const double tolerance =
        std::max(merge_tolerance, std::numeric_limits<double>::epsilon());
    std::unordered_set<uint64_t> seen_pairs;
    if (reserve_pairs != std::numeric_limits<size_t>::max()) {
      seen_pairs.reserve(reserve_pairs);
    }

    for (const auto &tile_pairs : tile_results) {
      for (const auto &pair : tile_pairs) {
        const uint64_t hash = hashPair(pair, tolerance);
        if (seen_pairs.insert(hash).second) {
          merged.push_back(pair);
        }
      }
    }

    // Stable order makes streaming results deterministic across runs.
    std::ranges::sort(merged, {}, [](const Pair &p) {
      return std::tuple(p.dimension, p.birth, p.death);
    });

    return merged;
  }

private:
  [[nodiscard]] static uint64_t hashPair(const Pair &pair, double tolerance) {
    const int64_t birth_q = quantize(pair.birth, tolerance);
    const int64_t death_q = quantize(pair.death, tolerance);

    uint64_t hash = static_cast<uint64_t>(pair.dimension);
    hash = hash * 1000003u + static_cast<uint64_t>(birth_q);
    hash = hash * 1000003u + static_cast<uint64_t>(death_q);
    return hash;
  }

  [[nodiscard]] static int64_t quantize(double value, double tolerance) {
    if (std::isnan(value)) {
      return 0;
    }
    if (value == std::numeric_limits<double>::infinity()) {
      return std::numeric_limits<int64_t>::max();
    }
    if (value == -std::numeric_limits<double>::infinity()) {
      return std::numeric_limits<int64_t>::min();
    }
    const long double scaled =
        static_cast<long double>(value) / static_cast<long double>(tolerance);
    if (scaled >=
        static_cast<long double>(std::numeric_limits<int64_t>::max())) {
      return std::numeric_limits<int64_t>::max();
    }
    if (scaled <=
        static_cast<long double>(std::numeric_limits<int64_t>::min())) {
      return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(scaled);
  }
};

[[nodiscard]] size_t resolveThreadCount(size_t configured_threads) {
  if (configured_threads > 0) {
    return configured_threads;
  }
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<size_t>(hw);
}

} // namespace
