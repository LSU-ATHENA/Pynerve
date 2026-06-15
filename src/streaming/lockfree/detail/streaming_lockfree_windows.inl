/**
 * @brief Memory-mapped file for large streaming datasets
 *
 * Enables out-of-core streaming of large point clouds
 */
class MemoryMappedFile {
public:
  MemoryMappedFile() : fd_(-1), data_(nullptr), size_(0) {}

  ~MemoryMappedFile() { close(); }

  bool open(const std::string &filename) {
    close();
    current_offset_ = 0;
    fd_ = ::open(filename.c_str(), O_RDONLY);
    if (fd_ < 0)
      return false;

    struct stat st;
    if (fstat(fd_, &st) < 0 || st.st_size < 0) {
      ::close(fd_);
      fd_ = -1;
      return false;
    }

    size_ = static_cast<size_t>(st.st_size);
    if (size_ == 0) {
      ::close(fd_);
      fd_ = -1;
      return true;
    }

    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
      ::close(fd_);
      fd_ = -1;
      data_ = nullptr;
      return false;
    }

    // Advise sequential access
    madvise(data_, size_, MADV_SEQUENTIAL);

    return true;
  }

  void close() {
    if (data_) {
      munmap(data_, size_);
      data_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    size_ = 0;
    current_offset_ = 0;
  }

  const char *data() const { return static_cast<const char *>(data_); }
  size_t size() const { return size_; }

  // Get chunk of data (advances internal pointer)
  std::vector<char> readChunk(size_t chunk_size) {
    std::vector<char> chunk;
    if (data_ == nullptr || current_offset_ >= size_) {
      return chunk;
    }
    chunk_size = std::min(chunk_size, size_ - current_offset_);

    if (chunk_size > 0) {
      chunk.resize(chunk_size);
      std::memcpy(chunk.data(),
                  static_cast<const char *>(data_) + current_offset_,
                  chunk_size);
      current_offset_ += chunk_size;
    }

    return chunk;
  }

private:
  int fd_;
  void *data_;
  size_t size_;
  size_t current_offset_ = 0;
};

/**
 * @brief Batched producer-consumer for high throughput
 *
 * Reduces synchronization overhead by batching
 */
template <typename T> class BatchedQueue {
public:
  explicit BatchedQueue(size_t batch_size = 64)
      : batch_size_(std::max<size_t>(batch_size, 1)),
        item_queue_(batchQueueCapacity(batch_size_)) {}

  // Producer: add item to batch
  void push(const T &item) {
    producer_batch_.push_back(item);

    if (producer_batch_.size() >= batch_size_) {
      flush();
    }
  }

  // Flush current batch
  void flush() {
    for (const auto &item : producer_batch_) {
      while (!item_queue_.push(item)) {
        // Spin or yield
        std::this_thread::yield();
      }
    }
    producer_batch_.clear();
  }

  // Consumer: get batch of items
  std::vector<T> popBatch() {
    std::vector<T> batch;
    batch.reserve(batch_size_);

    for (size_t i = 0; i < batch_size_; ++i) {
      auto item = item_queue_.pop();
      if (item) {
        batch.push_back(*item);
      } else {
        break;
      }
    }

    return batch;
  }

private:
  static size_t batchQueueCapacity(size_t batch_size) {
    if (batch_size > std::numeric_limits<size_t>::max() / 4) {
      throw std::length_error("batched lockfree queue capacity exceeds size_t");
    }
    return batch_size * 4;
  }

  size_t batch_size_;
  LockFreeSPSCQueue<T> item_queue_;
  std::vector<T> producer_batch_;
};

/**
 * @brief Streaming window manager
 *
 * Coordinates data ingestion, window updates, and persistence computation
 */
template <typename T> class StreamingWindowManager {
public:
  StreamingWindowManager(size_t window_size, size_t buffer_capacity)
      : window_size_(std::max<size_t>(window_size, 1)),
        circular_buffer_(std::max<size_t>(buffer_capacity, 1)),
        data_queue_(std::max<size_t>(buffer_capacity, 1)), running_(false) {}

  virtual ~StreamingWindowManager() { stop(); }

  void start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel)) {
      return;
    }

    // Start ingestion thread
    ingestion_thread_ = std::thread([this]() {
      while (running_.load(std::memory_order_acquire)) {
        auto data = data_queue_.pop();
        if (data) {
          circular_buffer_.write(*data);
        }
      }
    });

    // Start computation thread
    computation_thread_ = std::thread([this]() {
      while (running_.load(std::memory_order_acquire)) {
        auto window = circular_buffer_.getWindow(window_size_);
        if (window.size() >= window_size_) {
          processWindow(window);
        }
      }
    });
  }

  void stop() {
    running_.store(false, std::memory_order_release);
    if (ingestion_thread_.joinable()) {
      ingestion_thread_.join();
    }
    if (computation_thread_.joinable()) {
      computation_thread_.join();
    }
  }

  void ingest(const T &data) {
    while (!data_queue_.push(data)) {
      std::this_thread::yield();
    }
  }

  virtual void processWindow(const std::vector<T> &window) = 0;

private:
  size_t window_size_;
  CircularBuffer<T> circular_buffer_;
  LockFreeSPSCQueue<T> data_queue_;

  std::atomic<bool> running_;
  std::thread ingestion_thread_;
  std::thread computation_thread_;
};
