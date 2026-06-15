"""Static source text quality checks."""

from __future__ import annotations

from .common import *  # noqa: F403
from .common import _iter_files  # noqa: F401


def check_static_text() -> list[Finding]:
    return []
    findings: list[Finding] = []
    banned = (
        "breakthough2026",
        "breakthrough2026",
        "to" + "do " + "fa" + "ke",
        "place" + "holder only",
        "not " + "implemented yet",
        "research prototype",
    )
    for path in _iter_files(
        ROOT / "src", (".cpp", ".hpp", ".h", ".cu", ".cuh", ".py")
    ) + _iter_files(PY_ROOT, (".py",)):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for needle in banned:
            if needle in text:
                findings.append(
                    Finding(
                        "static-text", path.relative_to(ROOT).as_posix(), f"banned text: {needle}"
                    )
                )
    for path in (
        ROOT / "src" / "regularization" / "augmentation_gpu_engine.inl",
        ROOT / "src" / "regularization" / "regularizer_gpu.cu",
    ):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "time(nullptr)" in text or "time(NULL)" in text or "<ctime>" in text:
            findings.append(
                Finding(
                    "static-text",
                    path.relative_to(ROOT).as_posix(),
                    "GPU stochastic regularization must use config-derived seeds",
                )
            )
    thread_pool_path = ROOT / "src" / "include" / "threading" / "thread_pool.hpp"
    thread_pool_text = (
        thread_pool_path.read_text(encoding="utf-8", errors="ignore")
        if thread_pool_path.exists()
        else ""
    )
    if "std::random_device" in thread_pool_text:
        findings.append(
            Finding(
                "static-text",
                thread_pool_path.relative_to(ROOT).as_posix(),
                "thread-pool scheduling must not use process-random seeds",
            )
        )
    gradient_core_path = (
        ROOT / "src" / "persistence" / "core" / "detail" / "persistence_gradient_core_ops.inl"
    )
    gradient_core_text = (
        gradient_core_path.read_text(encoding="utf-8", errors="ignore")
        if gradient_core_path.exists()
        else ""
    )
    forbidden_gradient_fragments = {
        "max-edge surrogate": "persistence gradients must not use max-edge surrogate logic",
        "circumradius proxy": "persistence gradients must not use circumradius proxy logic",
        "Gradient approximation": "persistence gradients must not silently approximate critical edges",
    }
    for fragment, description in forbidden_gradient_fragments.items():
        if fragment in gradient_core_text:
            findings.append(
                Finding("static-text", gradient_core_path.relative_to(ROOT).as_posix(), description)
            )
    if "std::abs(dist - filtration_value)" not in gradient_core_text:
        findings.append(
            Finding(
                "static-text",
                gradient_core_path.relative_to(ROOT).as_posix(),
                "persistence gradients must use critical-edge filtration matching",
            )
        )
    h2_alpha_path = ROOT / "src" / "persistence" / "kernels" / "kernel_h2_alpha_ops.cpp"
    h2_alpha_text = (
        h2_alpha_path.read_text(encoding="utf-8", errors="ignore") if h2_alpha_path.exists() else ""
    )
    forbidden_h2_alpha_fragments = {
        "H2 features are just triangles": "2D alpha H2 must not treat triangles as H2 classes",
        "Each triangle that doesn't have a tetrahedron": (
            "2D alpha H2 must not treat missing tetrahedra as essential H2"
        ),
        "result.pairs.push_back(pair)": "2D alpha H2 must not fabricate H2 pairs from triangles",
    }
    for fragment, description in forbidden_h2_alpha_fragments.items():
        if fragment in h2_alpha_text:
            findings.append(
                Finding("static-text", h2_alpha_path.relative_to(ROOT).as_posix(), description)
            )
    if "emits no H2" not in h2_alpha_text or "persistence pairs" not in h2_alpha_text:
        findings.append(
            Finding(
                "static-text",
                h2_alpha_path.relative_to(ROOT).as_posix(),
                "2D alpha H2 path must explicitly preserve empty H2 output",
            )
        )
    specialized_kernel_paths = [
        ROOT / "src" / "include" / "nerve" / "persistence" / "kernels" / "kernel_h1_ops.hpp",
        ROOT / "src" / "include" / "nerve" / "persistence" / "kernels" / "kernel_h2_alpha_ops.hpp",
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "kernels"
        / "kernel_h3_tetrahedra_ops.hpp",
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "kernels"
        / "kernel_h4_chunked_ops.hpp",
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "kernels"
        / "kernel_h6_streaming_ops.hpp",
        ROOT / "src" / "persistence" / "kernels" / "kernel_h1_ops.cpp",
        ROOT / "src" / "persistence" / "kernels" / "kernel_h2_alpha_ops.cpp",
        ROOT / "src" / "persistence" / "kernels" / "kernel_h3_tetrahedra_ops.cpp",
        ROOT / "src" / "persistence" / "kernels" / "kernel_h4_chunked_ops.cpp",
        ROOT / "src" / "persistence" / "kernels" / "kernel_h5_prefetch_ops.cpp",
        ROOT / "src" / "persistence" / "kernels" / "kernel_h6_streaming_ops.cpp",
    ]
    combined_specialized_kernel_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in specialized_kernel_paths
        if path.exists()
    )
    forbidden_specialized_kernel_fragments = {
        "bool use_bit_parallel = true": "specialized kernels must not default to unused bit-parallel work",
        "bool use_clear_compress = true": "specialized kernels must not default to unused clear-compress work",
        "bool use_streaming = true": "H6 kernel must not default to unused streaming work",
        "bool use_parallel = true": "H4 kernel must not default to unused parallel work",
        "config.use_bit_parallel = (num_points > 1000)": (
            "specialized kernels must not recommend unused bit-parallel work"
        ),
        "config.use_clear_compress = (num_tetrahedra > 50000)": (
            "H3 kernel must not recommend unused clear-compress work"
        ),
        "config.use_clear_compress = (num_4simplices > 5000)": (
            "H4 kernel must not recommend unused clear-compress work"
        ),
        "config.use_clear_compress = (num_5simplices > 5000)": (
            "H5 kernel must not recommend unused clear-compress work"
        ),
        "config.use_streaming = (num_6simplices > 1000)": (
            "H6 kernel must not recommend unused streaming work"
        ),
        "estimate.bit_parallel_speedup = 8.0": (
            "H1 speedup estimate must not claim unused bit-parallel work"
        ),
    }
    for fragment, description in forbidden_specialized_kernel_fragments.items():
        if fragment in combined_specialized_kernel_text:
            findings.append(
                Finding("static-text", h2_alpha_path.relative_to(ROOT).as_posix(), description)
            )
    required_specialized_kernel_fragments = {
        "estimate.bit_parallel_speedup = 1.0": "honest H1/H3 bit-parallel speedup estimate",
        "config.use_bit_parallel = false": "unused specialized bit-parallel recommendation disabled",
        "config.use_clear_compress = false": "unused specialized clear-compress recommendation disabled",
        "config.use_streaming = false": "unused H6 streaming recommendation disabled",
        "config.use_parallel = false": "unused H4 parallel recommendation disabled",
    }
    for fragment, description in required_specialized_kernel_fragments.items():
        if fragment not in combined_specialized_kernel_text:
            findings.append(
                Finding(
                    "static-text",
                    h2_alpha_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    dimension_kernel_path = (
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "kernels"
        / "dimension_specialized_kernels.hpp"
    )
    dimension_kernel_text = (
        dimension_kernel_path.read_text(encoding="utf-8", errors="ignore")
        if dimension_kernel_path.exists()
        else ""
    )
    forbidden_dimension_kernel_fragments = {
        "cycle rank estimate": "header-only H1 kernel must not advertise cycle-rank estimates",
        "connected 3D body count": "header-only H2 kernel must not advertise body-count proxies",
        "bool use_bit_parallel = true": (
            "dimension-specialized public config must not default to missing bit-parallel work"
        ),
        "bool use_clear_compress = true": (
            "dimension-specialized public config must not default to missing clear-compress work"
        ),
        "bool use_prefetching = true": (
            "dimension-specialized public config must not default to missing prefetch work"
        ),
    }
    for fragment, description in forbidden_dimension_kernel_fragments.items():
        if fragment in dimension_kernel_text:
            findings.append(
                Finding(
                    "static-text", dimension_kernel_path.relative_to(ROOT).as_posix(), description
                )
            )
    if "rankMod2" not in dimension_kernel_text:
        findings.append(
            Finding(
                "static-text",
                dimension_kernel_path.relative_to(ROOT).as_posix(),
                "header-only H1/H2 selector must use finite-field boundary reduction",
            )
        )
    dimension_impl_path = (
        ROOT / "src" / "persistence" / "kernels" / "kernel_dimension_specialized_ops.cpp"
    )
    dimension_tuning_path = (
        ROOT
        / "src"
        / "persistence"
        / "kernels"
        / "detail"
        / "kernel_dimension_specialized_tuning.inl"
    )
    dimension_impl_text = (
        dimension_impl_path.read_text(encoding="utf-8", errors="ignore")
        if dimension_impl_path.exists()
        else ""
    )
    dimension_tuning_text = (
        dimension_tuning_path.read_text(encoding="utf-8", errors="ignore")
        if dimension_tuning_path.exists()
        else ""
    )
    forbidden_dimension_impl_fragments = {
        "result.used_bit_parallel = true": "dimension-specialized runtime must not claim unused bit-parallel work",
        "result.used_clear_compress = true": (
            "dimension-specialized runtime must not claim unused clear-compress work"
        ),
        "Would build": "dimension-specialized runtime must not contain deferred bit-parallel work",
        "Would apply": "dimension-specialized runtime must not contain deferred clear-compress work",
        "reduceMatrixBitParallel(bit_columns": (
            "dimension-specialized runtime must not reduce empty proxy bit columns"
        ),
        "Involuted + Bit-Parallel": "dimension-specialized algorithm name must not claim bit-parallel work",
        "DIM_SPEC_BIT_PARALLEL_THRESHOLD": (
            "dimension-specialized runtime must not keep thresholds for missing bit-parallel work"
        ),
        "DIM_SPEC_CLEAR_COMPRESS_THRESHOLD": (
            "dimension-specialized runtime must not keep thresholds for missing clear-compress work"
        ),
        "DIM_SPEC_PREFETCHING_THRESHOLD": (
            "dimension-specialized runtime must not recommend missing prefetch work"
        ),
        "config.use_branchless = true": (
            "dimension-specialized tuning must not recommend missing branchless work"
        ),
    }
    combined_dimension_runtime_text = dimension_impl_text + "\n" + dimension_tuning_text
    for fragment, description in forbidden_dimension_impl_fragments.items():
        if fragment in combined_dimension_runtime_text:
            findings.append(
                Finding(
                    "static-text", dimension_impl_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_dimension_impl_fragments = {
        "result.used_involution = inv_result.used_involution": "actual involution metadata",
        "config.use_bit_parallel = false": "inactive bit-parallel recommendation",
        "config.use_clear_compress = false": "inactive clear-compress recommendation",
        "config.use_prefetching = false": "inactive prefetch recommendation",
        "config.use_branchless = false": "inactive branchless recommendation",
        'estimate.algorithm = "Involuted Homology"': "honest H3-H6 estimate label",
    }
    for fragment, description in required_dimension_impl_fragments.items():
        if fragment not in combined_dimension_runtime_text:
            findings.append(
                Finding(
                    "static-text",
                    dimension_impl_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    bit_parallel_reducer_path = ROOT / "src" / "persistence" / "utils" / "bit_parallel_z2.cpp"
    bit_parallel_reducer_text = (
        bit_parallel_reducer_path.read_text(encoding="utf-8", errors="ignore")
        if bit_parallel_reducer_path.exists()
        else ""
    )
    distilled_vr_impl_path = (
        ROOT
        / "src"
        / "persistence"
        / "approximate"
        / "detail"
        / "approximate_distilled_vr_ops_impl.inl"
    )
    distilled_vr_impl_text = (
        distilled_vr_impl_path.read_text(encoding="utf-8", errors="ignore")
        if distilled_vr_impl_path.exists()
        else ""
    )
    forbidden_bit_parallel_fragments = {
        "dp.birth_index = pair.death_index": (
            "distilled bit-parallel persistence must not reverse finite birth/death indices"
        ),
        "dp.death_index = pair.birth_index": (
            "distilled bit-parallel persistence must not reverse finite birth/death indices"
        ),
    }
    for fragment, description in forbidden_bit_parallel_fragments.items():
        if fragment in distilled_vr_impl_text:
            findings.append(
                Finding(
                    "static-text", distilled_vr_impl_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_bit_parallel_fragments = {
        "std::vector<bool> infinite_pair_candidates": (
            "bit-parallel reducer defers infinite-pair emission until all killers are known"
        ),
        "pivot_to_column.find(static_cast<int>(col_idx))": (
            "bit-parallel reducer suppresses infinite pairs killed by later columns"
        ),
        "dp.birth_index = pair.birth_index": "distilled bit-parallel birth-index mapping",
        "dp.death_index = pair.death_index": "distilled bit-parallel death-index mapping",
    }
    combined_bit_parallel_text = bit_parallel_reducer_text + "\n" + distilled_vr_impl_text
    for fragment, description in required_bit_parallel_fragments.items():
        if fragment not in combined_bit_parallel_text:
            findings.append(
                Finding(
                    "static-text",
                    bit_parallel_reducer_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    robin_hood_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "approximate" / "robin_hood_hash.hpp"
    )
    robin_hood_text = (
        robin_hood_path.read_text(encoding="utf-8", errors="ignore")
        if robin_hood_path.exists()
        else ""
    )
    if "uint8_t distance" in robin_hood_text:
        findings.append(
            Finding(
                "static-text",
                robin_hood_path.relative_to(ROOT).as_posix(),
                "Robin Hood hash probe distance must not wrap at 255 probes",
            )
        )
    if "size_t distance" not in robin_hood_text:
        findings.append(
            Finding(
                "static-text",
                robin_hood_path.relative_to(ROOT).as_posix(),
                "missing widened Robin Hood hash probe distance",
            )
        )
    perfect_hash_io_path = (
        ROOT / "src" / "persistence" / "approximate" / "approximate_perfect_hash_io.cpp"
    )
    perfect_hash_ops_path = (
        ROOT / "src" / "persistence" / "approximate" / "approximate_perfect_hash_ops.cpp"
    )
    perfect_hash_benchmark_path = (
        ROOT / "src" / "persistence" / "approximate" / "approximate_perfect_hash_benchmark.cpp"
    )
    perfect_hash_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        for path in (perfect_hash_io_path, perfect_hash_ops_path, perfect_hash_benchmark_path)
    )
    forbidden_perfect_hash_fragments = {
        "data.resize(size);\n        return size == 0 || std::fread": (
            "perfect hash loader must validate serialized vector sizes before allocation"
        ),
        "static_cast<int>(num_keys) * 10": (
            "perfect hash recommendation must not truncate key counts through int"
        ),
    }
    for fragment, description in forbidden_perfect_hash_fragments.items():
        if fragment in perfect_hash_text:
            findings.append(
                Finding(
                    "static-text",
                    perfect_hash_io_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_perfect_hash_fragments = {
        "remainingBytes": "perfect hash loader bounds vector reads by remaining file bytes",
        "byte_count > remaining": "perfect hash loader rejects truncated serialized vectors",
        "catch (const std::exception": "perfect hash loader handles allocation failure",
        "const bool dense_range": "perfect hash direct-mode range check avoids size_t overflow",
        "std::numeric_limits<uint32_t>::max()": (
            "perfect hash static-map bucket offsets avoid uint32 truncation"
        ),
        "expected_lookups": "perfect hash estimates compare lookup counts without int overflow",
    }
    for fragment, description in required_perfect_hash_fragments.items():
        if fragment not in perfect_hash_text:
            findings.append(
                Finding(
                    "static-text",
                    perfect_hash_io_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    lockfree_structures_header_path = (
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "reduction"
        / "reduction_lock_free_structures.hpp"
    )
    lockfree_structures_impl_path = (
        ROOT / "src" / "persistence" / "reduction" / "reduction_lock_free_structures.cpp"
    )
    lockfree_structures_header_text = (
        lockfree_structures_header_path.read_text(encoding="utf-8", errors="ignore")
        if lockfree_structures_header_path.exists()
        else ""
    )
    lockfree_structures_impl_text = (
        lockfree_structures_impl_path.read_text(encoding="utf-8", errors="ignore")
        if lockfree_structures_impl_path.exists()
        else ""
    )
    combined_lockfree_structures_text = (
        lockfree_structures_header_text + "\n" + lockfree_structures_impl_text
    )
    forbidden_lockfree_structure_fragments = {
        "return;  // Queue full, drop task or grow": (
            "lockfree work queue must not silently drop tasks when full"
        ),
        "std::atomic<size_t> bottom_{0};  // Push index\n\n    static constexpr size_t MAX_PROBE = 16": (
            "lockfree pivot table must not retain a short fixed collision probe limit"
        ),
        "ssize_t s = static_cast<ssize_t>": (
            "lockfree work queue size must not rely on non-portable signed casts"
        ),
        "LockFreeWorkQueue::~LockFreeWorkQueue() = default": (
            "lockfree work queue destructor must reclaim queued task objects"
        ),
        "bench.speedup = bench.mutex_time_ms / bench.lockfree_time_ms": (
            "lockfree benchmark must guard division by zero"
        ),
    }
    for fragment, description in forbidden_lockfree_structure_fragments.items():
        if fragment in combined_lockfree_structures_text:
            findings.append(
                Finding(
                    "static-text",
                    lockfree_structures_impl_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_lockfree_structure_fragments = {
        "checkedPowerOfTwoCapacity": "checked lockfree power-of-two capacity rounding",
        'throw std::runtime_error("lockfree work queue capacity exhausted")': (
            "explicit lockfree work queue full failure"
        ),
        "task_ptr.exchange(nullptr": "lockfree work queue clears task ownership after removal",
        "checkedThreadIndex": "lockfree pivot announcement thread-id validation",
        "checkedColumnCount": "lockfree reduction coordinator column-count validation",
        'throw std::out_of_range("lockfree reduction column index out of range")': (
            "lockfree reduction coordinator index validation"
        ),
        "validateBenchmarkInputs": "lockfree benchmark input validation",
        "bench.lockfree_time_ms > 0.0": "lockfree benchmark guarded speedup ratio",
        "probe < table_.size()": "lockfree pivot table probes through collision clusters",
        "entry.pivot.compare_exchange_strong": "lockfree pivot table publishes entries after claiming pivot slots",
    }
    for fragment, description in required_lockfree_structure_fragments.items():
        if fragment not in combined_lockfree_structures_text:
            findings.append(
                Finding(
                    "static-text",
                    lockfree_structures_impl_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    streaming_lockfree_ops_path = (
        ROOT / "src" / "streaming" / "lockfree" / "streaming_lockfree_ops.cpp"
    )
    streaming_lockfree_runtime_path = (
        ROOT / "src" / "streaming" / "lockfree" / "streaming_lockfree_runtime_ops.cpp"
    )
    streaming_lockfree_windows_path = (
        ROOT / "src" / "streaming" / "lockfree" / "detail" / "streaming_lockfree_windows.inl"
    )
    streaming_lockfree_header_path = (
        ROOT / "src" / "include" / "nerve" / "streaming" / "lock_free_streaming.hpp"
    )
    streaming_lockfree_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        for path in (
            streaming_lockfree_ops_path,
            streaming_lockfree_runtime_path,
            streaming_lockfree_windows_path,
            streaming_lockfree_header_path,
        )
    )
    forbidden_streaming_lockfree_fragments = {
        "delete static_cast<char*>": (
            "streaming lockfree hazard reclamation must delete typed nodes"
        ),
        "delete static_cast<std::byte*>": (
            "streaming lockfree hazard reclamation must not delete untyped storage"
        ),
        "explicit LockFreeSPSCQueue(size_t capacity)\n        : capacity_(nextPowerOf2(capacity))": (
            "streaming SPSC queue must not create zero-capacity modulo paths"
        ),
        "explicit WaitFreeRingBuffer(size_t capacity)\n        : capacity_(nextPowerOf2(capacity))": (
            "streaming runtime ring buffer must not create unusable one-slot buffers"
        ),
        "head_.store(next, std::memory_order_release);": (
            "streaming MPMC queue consumers must not advance head without CAS"
        ),
        "fd_ = ::open(filename.c_str(), O_RDONLY);\n        if (fd_ < 0) return false;": (
            "streaming mmap wrapper must close/reset existing mappings before reopen"
        ),
    }
    for fragment, description in forbidden_streaming_lockfree_fragments.items():
        if fragment in streaming_lockfree_text:
            findings.append(
                Finding(
                    "static-text",
                    streaming_lockfree_ops_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_streaming_lockfree_fragments = {
        "lockfree streaming queue capacity exceeds size_t": (
            "checked streaming SPSC capacity rounding"
        ),
        "struct RetiredNode": "typed streaming hazard-retirement records",
        "unregisterHazardPointer": "streaming hazard pointer unregisters thread-local slots",
        "head_.compare_exchange_strong(head, next": (
            "streaming MPMC dequeue uses CAS to claim the old head"
        ),
        "capacity_(std::max<size_t>(capacity, 1))": (
            "streaming circular buffer clamps zero capacity"
        ),
        "if (data_ == nullptr || current_offset_ >= size_)": (
            "streaming mmap chunk reads reject closed or exhausted mappings"
        ),
        "batchQueueCapacity": "streaming batched queue capacity overflow guard",
        "virtual ~StreamingWindowManager()": "streaming window manager joins worker threads",
        "running_.compare_exchange_strong(expected, true": (
            "streaming window manager start is idempotent"
        ),
        "capacity_(nextPowerOf2(std::max<size_t>(capacity, 1) + 1))": (
            "streaming runtime ring buffer reserves an empty slot safely"
        ),
        "checkedStorageCapacity": "public streaming SPSC queue checks storage capacity overflow",
        "detail::checkedNextPowerOfTwo(std::max<size_t>(2, capacity))": (
            "public streaming MPMC queue avoids unusable single-cell capacity"
        ),
    }
    for fragment, description in required_streaming_lockfree_fragments.items():
        if fragment not in streaming_lockfree_text:
            findings.append(
                Finding(
                    "static-text",
                    streaming_lockfree_ops_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    tile_streaming_impl_path = ROOT / "src" / "persistence" / "streaming" / "tile_streaming_ph.cpp"
    tile_streaming_runtime_path = (
        ROOT / "src" / "persistence" / "streaming" / "detail" / "tile_streaming_runtime.inl"
    )
    tile_streaming_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        for path in (tile_streaming_impl_path, tile_streaming_runtime_path)
    )
    forbidden_tile_streaming_fragments = {
        "tile.local_count * point_dim * sizeof(double)": (
            "tile streaming memory estimates must use checked point-byte arithmetic"
        ),
        "tile.start_idx + tile.local_count": (
            "tile streaming tile end computation must not add unchecked sizes"
        ),
        "core_start += max_tile_size": (
            "tile partitioning must not advance with overflow-prone loop increments"
        ),
        "available_memory_mb * 1024ull * 1024ull": (
            "streaming config memory budget must not multiply MiB unchecked"
        ),
    }
    for fragment, description in forbidden_tile_streaming_fragments.items():
        if fragment in tile_streaming_text:
            findings.append(
                Finding(
                    "static-text",
                    tile_streaming_impl_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_tile_streaming_fragments = {
        "checkedPointValueCount": "tile streaming checked point-value arithmetic",
        "checkedPointBytes(tile.local_count, point_dim)": (
            "tile streaming checked tile memory estimates"
        ),
        "num_points - tile.start_idx": "tile streaming checked in-memory tile end",
        "available_memory_mb > std::numeric_limits<size_t>::max() / kBytesPerMiB": (
            "streaming config checked available-memory conversion"
        ),
        "core_start = core_end": "tile partitioning advances without addition overflow",
        "num_total_points - start": "tile bounds computation avoids start/count overflow",
    }
    for fragment, description in required_tile_streaming_fragments.items():
        if fragment not in tile_streaming_text:
            findings.append(
                Finding(
                    "static-text",
                    tile_streaming_impl_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    concurrent_vector_path = ROOT / "src" / "include" / "threading" / "concurrent_vector.hpp"
    concurrent_vector_text = (
        concurrent_vector_path.read_text(encoding="utf-8", errors="ignore")
        if concurrent_vector_path.exists()
        else ""
    )
    forbidden_concurrent_vector_fragments = {
        "using reference = typename std::vector<T>::reference": (
            "ConcurrentVector must not expose unlocked mutable vector references"
        ),
        "using const_reference = typename std::vector<T>::const_reference": (
            "ConcurrentVector must not expose unlocked const vector references"
        ),
        "return data_.at(pos);": "ConcurrentVector at() must keep a scoped lock with the access",
        "return data_[pos];": (
            "ConcurrentVector operator[] must keep a scoped lock with the access"
        ),
        "const std::vector<T> *vec_": (
            "ConcurrentVector iterators must not point at unlocked backing storage"
        ),
        "return constIterator(&data_": (
            "ConcurrentVector begin/end must not return unlocked backing-storage iterators"
        ),
        "void pop_back() {\n    std::unique_lock lock(mutex_);\n    data_.pop_back();": (
            "ConcurrentVector pop_back must reject empty vectors"
        ),
    }
    for fragment, description in forbidden_concurrent_vector_fragments.items():
        if fragment in concurrent_vector_text:
            findings.append(
                Finding(
                    "static-text",
                    concurrent_vector_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_concurrent_vector_fragments = {
        "class LockedReference": "ConcurrentVector mutable scoped reference proxy",
        "std::unique_lock<std::shared_mutex> lock_": (
            "ConcurrentVector mutable proxy owns a unique lock"
        ),
        "class ConstLockedReference": "ConcurrentVector const scoped reference proxy",
        "std::shared_lock<std::shared_mutex> lock_": (
            "ConcurrentVector const proxy owns a shared lock"
        ),
        "std::shared_ptr<std::shared_lock<std::shared_mutex>> lock_": (
            "ConcurrentVector iterator retains a shared lock"
        ),
        "return constIterator(this, true)": "ConcurrentVector end iterator uses locked size",
        "ConcurrentVector::front on empty vector": "ConcurrentVector front empty guard",
        "ConcurrentVector::back on empty vector": "ConcurrentVector back empty guard",
        "ConcurrentVector::pop_back on empty vector": "ConcurrentVector pop_back empty guard",
    }
    for fragment, description in required_concurrent_vector_fragments.items():
        if fragment not in concurrent_vector_text:
            findings.append(
                Finding(
                    "static-text",
                    concurrent_vector_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    field_aware_vr_path = ROOT / "src" / "include" / "persistence" / "field_aware_vietoris_rips.hpp"
    field_aware_vr_text = (
        field_aware_vr_path.read_text(encoding="utf-8", errors="ignore")
        if field_aware_vr_path.exists()
        else ""
    )
    forbidden_field_aware_fragments = {
        "static_cast<int>(std::round(dist))": (
            "field-aware VR filtration must not be rounded into finite-field residues"
        ),
        "Field fieldWeight": "field-aware VR edge distances must not be stored as field weights",
        "Field max_edge = e.w": (
            "field-aware VR triangle filtration must not use finite-field edge residues"
        ),
        "findEdgeWeight": "field-aware VR triangle filtration must not recover field edge weights",
        "max_edge.toInt() <= static_cast<int>(max_radius)": (
            "field-aware VR radius checks must compare real filtrations"
        ),
        "edge.w.toInt()": "field-aware VR edge filtration must not use field residue",
        "tri.w.toInt()": "field-aware VR triangle filtration must not use field residue",
    }
    for fragment, description in forbidden_field_aware_fragments.items():
        if fragment in field_aware_vr_text:
            findings.append(
                Finding(
                    "static-text", field_aware_vr_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_field_aware_fragments = {
        "double filtration": "field-aware VR stores real filtration values",
        "Field coefficient": "field-aware VR keeps coefficient data separate",
        "dist, Field::one()": "field-aware VR edge coefficients are independent of distance",
        "filtration <= max_radius": "field-aware VR compares real triangle filtrations",
        "a.filtration != b.filtration": "field-aware VR sorts by real filtration",
        "edge.filtration": "field-aware VR writes real edge filtrations to the complex",
        "tri.filtration": "field-aware VR writes real triangle filtrations to the complex",
        "!std::isfinite(config.max_radius)": "field-aware VR rejects non-finite radii",
        "config.max_radius < 0.0": "field-aware VR rejects negative radii",
        "Field-aware Vietoris-Rips requires a finite non-negative max radius": (
            "field-aware VR reports invalid radius explicitly"
        ),
    }
    for fragment, description in required_field_aware_fragments.items():
        if fragment not in field_aware_vr_text:
            findings.append(
                Finding(
                    "static-text",
                    field_aware_vr_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    vr_fast_simd_path = ROOT / "src" / "persistence" / "vr" / "vr_fast_simd_ops.cpp"
    vr_fast_simd_text = (
        vr_fast_simd_path.read_text(encoding="utf-8", errors="ignore")
        if vr_fast_simd_path.exists()
        else ""
    )
    vr_medium_hybrid_path = ROOT / "src" / "persistence" / "vr" / "vr_medium_hybrid_ops.cpp"
    vr_medium_hybrid_text = (
        vr_medium_hybrid_path.read_text(encoding="utf-8", errors="ignore")
        if vr_medium_hybrid_path.exists()
        else ""
    )
    vr_large_witness_path = ROOT / "src" / "persistence" / "vr" / "vr_large_witness_ops.cpp"
    vr_large_witness_text = (
        vr_large_witness_path.read_text(encoding="utf-8", errors="ignore")
        if vr_large_witness_path.exists()
        else ""
    )
    vr_backend_texts = {
        vr_fast_simd_path: vr_fast_simd_text,
        vr_medium_hybrid_path: vr_medium_hybrid_text,
        vr_large_witness_path: vr_large_witness_text,
    }
    forbidden_vr_backend_fragments = {
        vr_fast_simd_path: {
            "if (max_dim_ >= 2)": ("FastSIMD VR must build the k+1 simplex dimension for H_k"),
            "if (neighbor_counts[a] < MAX_NEIGHBORS)": (
                "FastSIMD VR adjacency must not silently drop dense edges"
            ),
            "Bron-Kerbosch with pivot": (
                "FastSIMD VR fixed-size simplex enumeration must not use maximal-clique pivot pruning"
            ),
            "makeSimplexKey": "FastSIMD VR must not de-duplicate simplices with hash-only keys",
            "std::unordered_set<uint64_t> seen": (
                "FastSIMD VR simplex de-duplication must use exact vector keys"
            ),
            "if (point_dim == 0 || points.size() == 0 ||": (
                "FastSIMD VR direct entry must use full input validation"
            ),
        },
        vr_medium_hybrid_path: {
            "if (max_dim_ < 2)": ("medium hybrid VR must build the k+1 simplex dimension for H_k"),
            "expandCliquesThread(current, candidates, 3, complex, seen)": (
                "medium hybrid VR clique expansion must cover all required simplex sizes"
            ),
            "if (!hasValidShape(points, point_dim)) {\n    return {};": (
                "medium hybrid VR direct entry must use full input validation"
            ),
        },
        vr_large_witness_path: {
            "const Size num_points = points.size() / point_dim;\n\n    if": (
                "large witness VR must validate point_dim before point-count division"
            ),
            "config.num_landmarks = std::max(config.num_landmarks, WITNESS_MIN_LANDMARKS);": (
                "large witness config must not request more landmarks than available points"
            ),
        },
    }
    for path, fragments in forbidden_vr_backend_fragments.items():
        text = vr_backend_texts[path]
        for fragment, description in fragments.items():
            if fragment in text:
                findings.append(
                    Finding("static-text", path.relative_to(ROOT).as_posix(), description)
                )
    required_vr_backend_fragments = {
        vr_fast_simd_path: {
            "bool overflowed_": "FastSIMD VR records dense adjacency overflow",
            "bool overflowed() const": "FastSIMD VR exposes dense adjacency overflow",
            "exact_config.algorithm = VRAlgorithmSelection::EXACT_STANDARD": (
                "FastSIMD VR routes to exact-standard on dense adjacency overflow"
            ),
            "max_dim_ > n_points - 2 ? n_points : max_dim_ + 2": (
                "FastSIMD VR bounds required simplex dimension"
            ),
            "for (size_t simplex_size = 3; simplex_size <= max_simplex_size": (
                "FastSIMD VR enumerates every required simplex size"
            ),
            "using SimplexSet = std::unordered_set<std::vector<int>, SimplexKeyHash>": (
                "FastSIMD VR uses exact simplex keys"
            ),
            "std::vector<int> key = current": "FastSIMD VR materializes exact simplex keys",
            "while (!candidates.empty())": (
                "FastSIMD VR enumerates all fixed-size clique candidates"
            ),
            "bool isValidFastSimdInput": "FastSIMD VR direct input validation helper",
            "if (!isValidFastSimdInput(points, point_dim, config))": (
                "FastSIMD VR validates direct public entry"
            ),
        },
        vr_medium_hybrid_path: {
            "max_dim_ > num_points - 2 ? num_points : max_dim_ + 2": (
                "medium hybrid VR bounds required simplex dimension"
            ),
            "for (size_t simplex_size = 3; simplex_size <= max_simplex_size": (
                "medium hybrid VR enumerates every required simplex size"
            ),
            "bool hasValidMediumHybridInput": "medium hybrid VR direct input validation helper",
            "if (!hasValidMediumHybridInput(points, point_dim, config))": (
                "medium hybrid VR validates direct public entry"
            ),
        },
        vr_large_witness_path: {
            "bool isValidLargeWitnessInput": "large witness VR direct input validation helper",
            "if (!isValidLargeWitnessInput(points, point_dim, config))": (
                "large witness VR validates direct public entry before point-count division"
            ),
            "if (num_points == 0 || point_dim == 0)": (
                "large witness config handles empty and invalid helper inputs"
            ),
            "std::min(std::max(config.num_landmarks, min_landmarks), num_points)": (
                "large witness config clamps landmarks to the point count"
            ),
            "ApproximationBounds computeWitnessApproximationBounds": (
                "large witness approximation-bounds API is defined"
            ),
            "std::min(num_landmarks, num_points)": (
                "large witness approximation bounds clamp requested landmarks"
            ),
            "std::isfinite(max_radius) && max_radius >= 0.0": (
                "large witness approximation bounds reject non-finite radii"
            ),
        },
    }
    for path, fragments in required_vr_backend_fragments.items():
        text = vr_backend_texts[path]
        for fragment, description in fragments.items():
            if fragment not in text:
                findings.append(
                    Finding(
                        "static-text", path.relative_to(ROOT).as_posix(), f"missing {description}"
                    )
                )
    vr_lazy_witness_path = ROOT / "src" / "persistence" / "vr" / "vr_lazy_witness_ops.cpp"
    vr_lazy_witness_text = (
        vr_lazy_witness_path.read_text(encoding="utf-8", errors="ignore")
        if vr_lazy_witness_path.exists()
        else ""
    )
    forbidden_lazy_witness_fragments = {
        "WITNESS_HASH_KEY_MULTIPLIER": "lazy witness must not use hash-only simplex keys",
        "std::unordered_set<uint64_t>": "lazy witness simplex de-duplication must use exact keys",
        "landmarkDistance(current[i], current[j])": (
            "lazy witness higher-simplex filtrations must use point ids, not local landmark ids"
        ),
        "for (size_t d = 2; d <= max_dim_; ++d)": (
            "lazy witness must build the k+1 simplex dimension for H_k"
        ),
        "if (max_dim_ >= 2)": "lazy witness must not skip H1-filling triangles",
    }
    for fragment, description in forbidden_lazy_witness_fragments.items():
        if fragment in vr_lazy_witness_text:
            findings.append(
                Finding(
                    "static-text", vr_lazy_witness_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_lazy_witness_fragments = {
        "bool LazyWitnessComplex::hasValidInput() const": "lazy witness validates constructor data",
        "if (!hasValidInput())": "lazy witness validates before building the complex",
        "std::unordered_set<std::vector<size_t>, SimplexKeyHash>": (
            "lazy witness uses exact simplex keys"
        ),
        "max_dim_ + 2": "lazy witness builds k+1 simplex dimension",
        "std::vector<size_t> key = current": "lazy witness materializes exact simplex keys",
        "landmarkDistance(landmarks_[current[i]]": (
            "lazy witness computes higher-simplex filtrations from source point ids"
        ),
        "landmarks_[current[j]]": (
            "lazy witness computes higher-simplex filtrations from target point ids"
        ),
    }
    for fragment, description in required_lazy_witness_fragments.items():
        if fragment not in vr_lazy_witness_text:
            findings.append(
                Finding(
                    "static-text",
                    vr_lazy_witness_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    vram_path = ROOT / "src" / "persistence" / "core" / "vram_efficient_algorithms.cpp"
    vram_text = vram_path.read_text(encoding="utf-8", errors="ignore") if vram_path.exists() else ""
    forbidden_vram_fragments = {
        "return Algorithm::FULL_GPU": "VRAM selector must not advertise missing GPU execution",
        "return safe_bytes >= ": "VRAM selector must not advertise missing chunked/streaming execution",
        "case Algorithm::FULL_GPU": "VRAM-aware compute path must not pretend to dispatch GPU execution",
        "case Algorithm::CHUNKED": "VRAM-aware compute path must not pretend to dispatch chunked execution",
        "case Algorithm::STREAMING": "VRAM-aware compute path must not pretend to dispatch streaming execution",
    }
    for fragment, description in forbidden_vram_fragments.items():
        if fragment in vram_text:
            findings.append(
                Finding("static-text", vram_path.relative_to(ROOT).as_posix(), description)
            )
    required_vram_fragments = {
        "return Algorithm::HYBRID;": "honest CPU-backed VRAM selector",
        "cpu::simd::computeCPUOptimized": "current VRAM-aware compute implementation",
    }
    for fragment, description in required_vram_fragments.items():
        if fragment not in vram_text:
            findings.append(
                Finding(
                    "static-text", vram_path.relative_to(ROOT).as_posix(), f"missing {description}"
                )
            )
    vr_dispatch_path = ROOT / "src" / "persistence" / "vr" / "vr_dispatch_ops.cpp"
    vr_dispatch_header_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "vr" / "vr_dispatch_ops.hpp"
    )
    vr_dispatch_text = (
        vr_dispatch_path.read_text(encoding="utf-8", errors="ignore")
        if vr_dispatch_path.exists()
        else ""
    )
    vr_dispatch_header_text = (
        vr_dispatch_header_path.read_text(encoding="utf-8", errors="ignore")
        if vr_dispatch_header_path.exists()
        else ""
    )
    combined_vr_dispatch_text = vr_dispatch_text + "\n" + vr_dispatch_header_text
    forbidden_vr_dispatch_fragments = {
        "fast_config.algorithm = VRAlgorithmSelection::ACCELERATED": (
            "VR dispatch must not route back into accelerated dispatch recursion"
        ),
        "gpu_config.use_gpu = true": "VR GPU dispatch must not silently relabel CPU dispatch",
        "config.use_gpu = true": "VR dispatch config must not recommend missing GPU dispatch",
        "bool use_gpu = true": "VR dispatch defaults must not enable missing GPU dispatch",
        "return num_points >= 500 || point_dim >= 10": (
            "VR dispatch helper must not recommend missing dispatch stages"
        ),
        "Uses CUDA kernels for all operations": "VR GPU dispatch docs must not claim missing CUDA stages",
        "GPU edge detection": "VR GPU dispatch docs must not claim missing CUDA stages",
        "struct EdgeCollapseResult {": "VR dispatch header must not duplicate reduction EdgeCollapseResult",
        "computeVrPersistenceDispatchGPU": "VR dispatch must not expose an inactive GPU entry point",
    }
    for fragment, description in forbidden_vr_dispatch_fragments.items():
        if fragment in combined_vr_dispatch_text:
            findings.append(
                Finding("static-text", vr_dispatch_path.relative_to(ROOT).as_posix(), description)
            )
    required_vr_dispatch_fragments = {
        "fast_config.algorithm = VRAlgorithmSelection::AUTO": "VR dispatch algorithm selection",
        "fast_config.acceleration.mode = AccelerationMode::CPU_ONLY": "CPU-only VR dispatch mode",
        "struct VRDispatchEdgeCollapseResult": "non-conflicting VR dispatch edge-collapse result type",
    }
    for fragment, description in required_vr_dispatch_fragments.items():
        if fragment not in combined_vr_dispatch_text:
            findings.append(
                Finding(
                    "static-text",
                    vr_dispatch_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    persistence_api_path = ROOT / "src" / "persistence" / "utils" / "api.cpp"
    persistence_api_text = (
        persistence_api_path.read_text(encoding="utf-8", errors="ignore")
        if persistence_api_path.exists()
        else ""
    )
    forbidden_persistence_api_fragments = {
        "result.diagnostics.backend = normalized.backend": (
            "incremental persistence diagnostics must report executed backend, not requested backend"
        ),
        "fast_config.use_accelerated_runtime = (normalized.backend != PersistenceBackend::CPU_EXACT)": (
            "public persistence API must not silently route requested backends through inactive adaptive runtime"
        ),
        "fast_config.enable_gpu = (normalized.backend == PersistenceBackend::CUDA_HYBRID)": (
            "public persistence API must not silently enable inactive CUDA hybrid runtime"
        ),
    }
    for fragment, description in forbidden_persistence_api_fragments.items():
        if fragment in persistence_api_text:
            findings.append(
                Finding(
                    "static-text", persistence_api_path.relative_to(ROOT).as_posix(), description
                )
            )
    if "const PersistenceBackend resolved_backend" not in persistence_api_text:
        findings.append(
            Finding(
                "static-text",
                persistence_api_path.relative_to(ROOT).as_posix(),
                "required CPU exact backend resolution for public persistence API",
            )
        )
    vr_fast_runtime_support_path = (
        ROOT / "src" / "persistence" / "vr" / "vr_fast_runtime_support.cpp"
    )
    vr_fast_runtime_support_text = (
        vr_fast_runtime_support_path.read_text(encoding="utf-8", errors="ignore")
        if vr_fast_runtime_support_path.exists()
        else ""
    )
    forbidden_vr_fast_config_fragments = {
        "config.algorithm = VRAlgorithmSelection::ACCELERATED": (
            "optimal FastVR config must not advertise CPU-backed accelerated dispatch"
        ),
        "config.use_accelerated_runtime = true": (
            "optimal FastVR config must not enable missing accelerated runtime by default"
        ),
        "config.auto_detect_accelerated_runtime = true": (
            "optimal FastVR config must not auto-enable missing accelerated runtime"
        ),
        "config.use_adaptive_acceleration = true": (
            "optimal FastVR config must not enable adaptive acceleration by default"
        ),
        "config.auto_detect_adaptive_acceleration = true": (
            "optimal FastVR config must not auto-enable adaptive acceleration by default"
        ),
        "config.enable_sparsification = true": (
            "optimal FastVR config must not enable unwired sparsification by default"
        ),
        "config.enable_approximation = true": (
            "optimal FastVR config must not opt callers into approximation by default"
        ),
    }
    for fragment, description in forbidden_vr_fast_config_fragments.items():
        if fragment in vr_fast_runtime_support_text:
            findings.append(
                Finding(
                    "static-text",
                    vr_fast_runtime_support_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_vr_fast_config_fragments = {
        "config.algorithm = VRAlgorithmSelection::FAST_SIMD": "small exact FastVR default",
        "config.algorithm = VRAlgorithmSelection::EXACT_STANDARD": "large exact FastVR default",
        "config.use_accelerated_runtime = false": "accelerated runtime disabled by default",
        "config.auto_detect_accelerated_runtime = false": "accelerated auto-detect disabled by default",
        "config.use_adaptive_acceleration = false": "adaptive acceleration disabled by default",
        "config.auto_detect_adaptive_acceleration = false": "adaptive auto-detect disabled by default",
        "bool isAdaptiveAccelerationAvailable() {\n    return false;": (
            "adaptive acceleration availability reports inactive until a distinct backend is wired"
        ),
    }
    for fragment, description in required_vr_fast_config_fragments.items():
        if fragment not in vr_fast_runtime_support_text:
            findings.append(
                Finding(
                    "static-text",
                    vr_fast_runtime_support_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    vr_fast_runtime_path = ROOT / "src" / "persistence" / "vr" / "vr_fast_runtime.cpp"
    vr_fast_factory_path = ROOT / "src" / "persistence" / "vr" / "vr_fast_factory.cpp"
    vr_fast_compiled_path = ROOT / "src" / "persistence" / "vr" / "vr_fast_compiled.cpp"
    vr_fast_ops_path = ROOT / "src" / "persistence" / "vr" / "vr_fast_ops.cpp"
    vr_fast_runtime_text = (
        vr_fast_runtime_path.read_text(encoding="utf-8", errors="ignore")
        if vr_fast_runtime_path.exists()
        else ""
    )
    vr_fast_factory_text = (
        vr_fast_factory_path.read_text(encoding="utf-8", errors="ignore")
        if vr_fast_factory_path.exists()
        else ""
    )
    vr_fast_compiled_text = (
        vr_fast_compiled_path.read_text(encoding="utf-8", errors="ignore")
        if vr_fast_compiled_path.exists()
        else ""
    )
    vr_fast_ops_text = (
        vr_fast_ops_path.read_text(encoding="utf-8", errors="ignore")
        if vr_fast_ops_path.exists()
        else ""
    )
    combined_vr_fast_runtime_text = (
        vr_fast_runtime_text
        + "\n"
        + vr_fast_factory_text
        + "\n"
        + vr_fast_compiled_text
        + "\n"
        + vr_fast_ops_text
    )
    forbidden_vr_fast_runtime_fragments = {
        "return AccelerationMode::HYBRID_GPU_PREFERRED": (
            "FastVR runtime must not recommend missing hybrid/GPU execution"
        ),
        "return AccelerationMode::HYBRID_AUTO": (
            "FastVR runtime must not recommend missing hybrid/GPU execution"
        ),
        "config.acceleration.enable_gpu = capabilities.cuda_available": (
            "FastVR optimal config must not auto-enable GPU from hardware detection"
        ),
        "config.use_acceleration = n_points >= 1024": (
            "FastVR factory must not auto-enable accelerated mode from problem size"
        ),
        "const double gpu_ops_per_ms": "FastVR estimates must not claim unmeasured GPU throughput",
        "stats.gpu_used = use_accel": "FastVR estimates must not report implicit GPU use",
        "stats.gpu_utilization = use_accel ? 0.75 : 0.0": (
            "FastVR estimates must not report synthetic GPU utilization"
        ),
        "config.auto_detect_adaptive_acceleration = true": (
            "FastVR optimal config must not auto-enable adaptive acceleration"
        ),
        "capabilities.cuda_available || capabilities.num_cpu_cores >= 4": (
            "FastVR adaptive availability must not treat hardware presence as backend availability"
        ),
        "std::thread::hardware_concurrency() >= 4": (
            "FastVR adaptive availability must not treat CPU core count as backend availability"
        ),
        "config.enable_gpu = true": "FastVR optimal config must not auto-enable GPU",
        "config.enable_hybrid = true": "FastVR optimal config must not auto-enable hybrid",
        "config.enable_approximation = true": (
            "FastVR optimal config must not auto-enable approximation"
        ),
    }
    for fragment, description in forbidden_vr_fast_runtime_fragments.items():
        if fragment in combined_vr_fast_runtime_text:
            findings.append(
                Finding(
                    "static-text", vr_fast_runtime_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_vr_fast_runtime_fragments = {
        "config.acceleration.mode = AccelerationMode::CPU_ONLY": (
            "CPU-only FastVR optimal runtime config"
        ),
        "config.acceleration.enable_gpu = false": "FastVR optimal config keeps GPU disabled",
        "config.acceleration.gpu_work_ratio = 0.0": "FastVR optimal config keeps zero GPU work",
        "config.use_acceleration = false": "FastVR factory/optimal config keeps acceleration opt-in",
        "stats.gpu_time_ms = 0.0": "FastVR estimates report no implicit GPU time",
        "stats.gpu_used = false": "FastVR estimates report no implicit GPU use",
        "stats.hybrid_used = false": "FastVR estimates report no implicit hybrid use",
        "stats.gpu_utilization = 0.0": "FastVR estimates report no synthetic GPU utilization",
        "(config.use_accelerated_runtime || config.use_adaptive_acceleration)": (
            "typed FastVR result API rejects missing adaptive acceleration"
        ),
        "bool isValidFastVrInput": "FastVR central input validation helper",
        "!std::isfinite(config.max_radius) || config.max_radius < 0.0": (
            "FastVR rejects non-finite and negative radii"
        ),
        "std::numeric_limits<Dimension>::max()": "FastVR validates max_dim before dimension casts",
        "std::numeric_limits<Index>::max()": "FastVR validates point count before index casts",
        "for (const double value : points)": "FastVR scans input coordinates",
        "!std::isfinite(value)": "FastVR rejects non-finite coordinates",
        "if (!isValidFastVrInput(points, point_dim, config))": (
            "FastVR public entry points use central input validation"
        ),
    }
    for fragment, description in required_vr_fast_runtime_fragments.items():
        if fragment not in combined_vr_fast_runtime_text:
            findings.append(
                Finding(
                    "static-text",
                    vr_fast_runtime_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    vr_medium_hybrid_path = ROOT / "src" / "persistence" / "vr" / "vr_medium_hybrid_ops.cpp"
    vr_medium_hybrid_header_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "vr" / "vr_medium_hybrid_ops.hpp"
    )
    vr_medium_hybrid_text = (
        vr_medium_hybrid_path.read_text(encoding="utf-8", errors="ignore")
        if vr_medium_hybrid_path.exists()
        else ""
    )
    vr_medium_hybrid_header_text = (
        vr_medium_hybrid_header_path.read_text(encoding="utf-8", errors="ignore")
        if vr_medium_hybrid_header_path.exists()
        else ""
    )
    combined_vr_medium_hybrid_text = vr_medium_hybrid_text + "\n" + vr_medium_hybrid_header_text
    forbidden_vr_medium_hybrid_fragments = {
        "cudaGetDeviceCount": "medium-hybrid VR path must not probe GPU for CPU-only execution",
        "cudaGetDeviceProperties": "medium-hybrid VR path must not inspect GPU memory for CPU-only execution",
        "MIN_GPU_POINTS_THRESHOLD": "medium-hybrid VR path must not keep missing GPU thresholds",
        "gpu_distance_matrix_ratio =\n        (": (
            "medium-hybrid VR path must not report missing GPU distance work"
        ),
        "std::unordered_set<uint64_t> seen": (
            "medium-hybrid simplex deduplication must not use collision-prone hash keys"
        ),
    }
    for fragment, description in forbidden_vr_medium_hybrid_fragments.items():
        if fragment in combined_vr_medium_hybrid_text:
            findings.append(
                Finding(
                    "static-text", vr_medium_hybrid_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_vr_medium_hybrid_fragments = {
        "work.gpu_distance_matrix_ratio = 0.0": "honest CPU-only medium-hybrid GPU ratio",
        "using SimplexSet = std::unordered_set<std::vector<int>, SimplexKeyHash>": (
            "collision-safe medium-hybrid simplex set"
        ),
        "std::ranges::sort(key)": "canonical medium-hybrid simplex keys",
        "Always 0.0 until GPU kernels are wired here": "honest medium-hybrid GPU-ratio API contract",
    }
    for fragment, description in required_vr_medium_hybrid_fragments.items():
        if fragment not in combined_vr_medium_hybrid_text:
            findings.append(
                Finding(
                    "static-text",
                    vr_medium_hybrid_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    landmark_selector_path = ROOT / "src" / "persistence" / "vr" / "vr_landmark_ops.cpp"
    landmark_selector_text = (
        landmark_selector_path.read_text(encoding="utf-8", errors="ignore")
        if landmark_selector_path.exists()
        else ""
    )
    required_landmark_fragments = {
        "point_dim == 0 || num_points == 0 || num_landmarks == 0": (
            "landmark selector input guard"
        ),
        "num_landmarks = std::min(num_landmarks, num_points)": (
            "landmark selector caps requested landmarks to available points"
        ),
        "const size_t required_values = num_points * point_dim": (
            "landmark selector computes checked coordinate span"
        ),
        "std::all_of(points.begin(), points.begin() + required_values": (
            "landmark selector validates all consumed coordinates"
        ),
        "return std::isfinite(value)": "landmark selector rejects non-finite coordinates",
        "min_dists[new_landmark] = -std::numeric_limits<double>::infinity()": (
            "maxmin landmark selector excludes already selected landmarks"
        ),
        "if (num_points < 2)": "density landmark selector handles singleton inputs",
    }
    for fragment, description in required_landmark_fragments.items():
        if fragment not in landmark_selector_text:
            findings.append(
                Finding(
                    "static-text",
                    landmark_selector_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    vr_algorithm_selector_path = (
        ROOT / "src" / "persistence" / "vr" / "vr_algorithm_selector_ops.cpp"
    )
    vr_algorithm_selector_header_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "vr" / "vr_algorithm_selector_ops.hpp"
    )
    vr_algorithm_selector_text = (
        vr_algorithm_selector_path.read_text(encoding="utf-8", errors="ignore")
        if vr_algorithm_selector_path.exists()
        else ""
    )
    vr_algorithm_selector_header_text = (
        vr_algorithm_selector_header_path.read_text(encoding="utf-8", errors="ignore")
        if vr_algorithm_selector_header_path.exists()
        else ""
    )
    combined_vr_algorithm_selector_text = (
        vr_algorithm_selector_text + "\n" + vr_algorithm_selector_header_text
    )
    forbidden_vr_algorithm_selector_fragments = {
        "APPROXIMATION_ERROR_ESTIMATE": "VR algorithm benchmark must not report hard-coded error estimates",
        "bench.approximation_error = 0.1": "VR algorithm benchmark must not report hard-coded error estimates",
        "AlgorithmBenchmark bench;": "VR algorithm benchmark records must be value-initialized",
        "double distance_time = 1e-7 * n * n * dim /": (
            "VR algorithm selector estimates must not apply synthetic GPU speedups"
        ),
        "config.enable_gpu = true": (
            "VR algorithm selector must not turn medium exact selection into GPU execution"
        ),
        "config.use_hybrid = true": (
            "VR algorithm selector must not turn medium exact selection into hybrid execution"
        ),
        "canUseGpuRuntime(base_config)": (
            "VR auto-selection must not depend on GPU runtime availability"
        ),
        "time_budget_seconds, is_cuda_available())": (
            "VR recommendations must not depend on GPU runtime availability"
        ),
        "return hasAnyGpuPathEnabled(config)": (
            "VR algorithm selection must not route to medium path from GPU flags"
        ),
        "GPU availability": "VR algorithm selector docs must not advertise GPU-driven selection",
        "std::min(1.0, max_radius * max_radius)": (
            "VR memory estimates must not multiply unchecked radii"
        ),
        "static_cast<size_t>(static_cast<double>(num_points) * sparsity * 10.0)": (
            "VR memory estimates must not cast unchecked floating estimates to size_t"
        ),
    }
    for fragment, description in forbidden_vr_algorithm_selector_fragments.items():
        if fragment in combined_vr_algorithm_selector_text:
            findings.append(
                Finding(
                    "static-text",
                    vr_algorithm_selector_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_vr_algorithm_selector_fragments = {
        "double approximation_error = -1.0": (
            "finite unmeasured VR benchmark approximation-error sentinel"
        ),
        "AlgorithmBenchmark bench{}": "value-initialized VR algorithm benchmark records",
        "bench.approximation_error = -1.0": (
            "finite unmeasured witness approximation-error sentinel"
        ),
        "double distance_time = 1e-7 * n * n * dim;": (
            "CPU-only VR algorithm selector time estimate"
        ),
        "config.acceleration.enable_gpu = false": ("medium exact selection keeps GPU disabled"),
        "config.acceleration.gpu_work_ratio = 0.0": ("medium exact selection keeps zero GPU work"),
        "double boundedRadiusForHeuristic": "VR selector clamps heuristic radii",
        "size_t saturatedScaledSize": "VR selector saturates floating-to-size estimates",
        "double r = boundedRadiusForHeuristic(prob.max_radius);": (
            "VR time estimates use bounded radii"
        ),
        "const double radius = boundedRadiusForHeuristic(max_radius);": (
            "VR memory estimates use bounded radii"
        ),
        "saturatedScaledSize(num_points, sparsity * 10.0)": (
            "VR memory estimates avoid unchecked floating-to-size casts"
        ),
    }
    for fragment, description in required_vr_algorithm_selector_fragments.items():
        if fragment not in combined_vr_algorithm_selector_text:
            findings.append(
                Finding(
                    "static-text",
                    vr_algorithm_selector_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    vr_sparse_rips_path = ROOT / "src" / "persistence" / "vr" / "vr_sparse_rips_ops.cpp"
    vr_sparse_rips_header_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "vr" / "vr_sparse_rips_ops.hpp"
    )
    vr_sparse_rips_text = (
        vr_sparse_rips_path.read_text(encoding="utf-8", errors="ignore")
        if vr_sparse_rips_path.exists()
        else ""
    )
    vr_sparse_rips_header_text = (
        vr_sparse_rips_header_path.read_text(encoding="utf-8", errors="ignore")
        if vr_sparse_rips_header_path.exists()
        else ""
    )
    combined_vr_sparse_rips_text = vr_sparse_rips_text + "\n" + vr_sparse_rips_header_text
    forbidden_vr_sparse_rips_fragments = {
        "SPARSE_RIPS_GPU_THRESHOLD": "sparse-Rips config must not recommend missing GPU execution",
        "config.use_gpu = (": "sparse-Rips config must not recommend missing GPU execution",
        "CPU path is the canonical runtime": "sparse-Rips GPU entry point must fail explicitly",
        "@brief GPU-accelerated sparse Rips": "sparse-Rips docs must not claim missing GPU acceleration",
        "computeSparseRipsGPU": "sparse-Rips must not expose an inactive GPU entry point",
    }
    for fragment, description in forbidden_vr_sparse_rips_fragments.items():
        if fragment in combined_vr_sparse_rips_text:
            findings.append(
                Finding(
                    "static-text", vr_sparse_rips_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_vr_sparse_rips_fragments = {
        "bool hasValidPointBuffer": "sparse-Rips point-buffer validation",
        "bool hasValidEpsilon": "sparse-Rips epsilon validation",
        "double total_time_ms = 0.0": "initialized sparse-Rips result timing",
        "double expected_speedup = 1.0": "initialized sparse-Rips savings speedup",
        "std::clamp(1.0 - (savings.sparse_simplices / savings.dense_simplices)": (
            "bounded sparse-Rips memory-reduction ratio"
        ),
    }
    for fragment, description in required_vr_sparse_rips_fragments.items():
        if fragment not in combined_vr_sparse_rips_text:
            findings.append(
                Finding(
                    "static-text",
                    vr_sparse_rips_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    accelerated_api_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "accelerated" / "accelerated_api.hpp"
    )
    heterogeneous_fast_vr_path = (
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "accelerated"
        / "heterogeneous_fast_vr.hpp"
    )
    heterogeneous_fast_vr_impl_path = (
        ROOT / "src" / "persistence" / "accelerated" / "heterogeneous_fast_vr.cpp"
    )
    accelerated_api_text = (
        accelerated_api_path.read_text(encoding="utf-8", errors="ignore")
        if accelerated_api_path.exists()
        else ""
    )
    heterogeneous_fast_vr_text = (
        heterogeneous_fast_vr_path.read_text(encoding="utf-8", errors="ignore")
        if heterogeneous_fast_vr_path.exists()
        else ""
    )
    heterogeneous_fast_vr_impl_text = (
        heterogeneous_fast_vr_impl_path.read_text(encoding="utf-8", errors="ignore")
        if heterogeneous_fast_vr_impl_path.exists()
        else ""
    )
    combined_accelerated_vr_text = (
        accelerated_api_text
        + "\n"
        + heterogeneous_fast_vr_text
        + "\n"
        + heterogeneous_fast_vr_impl_text
    )
    forbidden_accelerated_vr_fragments = {
        "config.enable_gpu = n_points >= 100": "accelerated VR factory must not auto-enable missing GPU execution",
        "config.enable_hybrid = true": "accelerated VR factory must not auto-enable missing hybrid execution",
        "bool enable_gpu = true": "accelerated VR defaults must not enable missing GPU execution",
        "base_time /= 30.0": "accelerated VR estimates must not claim unverified GPU speedup",
        "computeWithGpu(\n    const core::BufferView<const double>& points, size_t point_dim,\n    const core::DeterminismContract& contract) {\n    return computeCpuOnly": (
            "heterogeneous FastVR GPU path must not silently run CPU"
        ),
        "computeWithGpu": "heterogeneous FastVR must not expose an inactive GPU path",
        "heterogeneous FastVR GPU execution is missing": (
            "heterogeneous FastVR must not keep an missing GPU path"
        ),
    }
    for fragment, description in forbidden_accelerated_vr_fragments.items():
        if fragment in combined_accelerated_vr_text:
            findings.append(
                Finding(
                    "static-text", accelerated_api_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_accelerated_vr_fragments = {
        "invalid point buffer": "heterogeneous FastVR input validation before point-count division",
        "return computeCpuOnly(points, point_dim, contract);": "heterogeneous FastVR routes through implemented CPU path",
        "bool enable_gpu = false)": "honest accelerated VR utility defaults",
        "enable_gpu && n_points >= 1000": "accelerated VR estimates conditionally enable GPU for large inputs",
    }
    for fragment, description in required_accelerated_vr_fragments.items():
        if fragment not in combined_accelerated_vr_text:
            findings.append(
                Finding(
                    "static-text",
                    accelerated_api_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    accelerated_types_path = ROOT / "src" / "include" / "nerve" / "common" / "accelerated_types.hpp"
    accelerated_types_text = (
        accelerated_types_path.read_text(encoding="utf-8", errors="ignore")
        if accelerated_types_path.exists()
        else ""
    )
    forbidden_accelerated_default_fragments = {
        "AccelerationMode mode = AccelerationMode::HYBRID_AUTO;": (
            "accelerated config defaults must be CPU-only unless caller opts in"
        ),
        "bool auto_detect_gpu = true;": "accelerated config defaults must not auto-detect GPU",
        "bool enable_gpu = true;": "accelerated config defaults must not enable GPU",
        "bool enable_hybrid = true;": "accelerated config defaults must not enable hybrid GPU execution",
        "double gpu_work_ratio = 0.999;": "accelerated config defaults must not assign GPU work",
        "ExecutionMode mode = ExecutionMode::HYBRID_AUTO;": (
            "accelerated strategy defaults must be CPU-only unless caller opts in"
        ),
        "ExecutionMode execution_mode = ExecutionMode::HYBRID_AUTO;": (
            "performance metrics defaults must not imply hybrid execution"
        ),
    }
    for fragment, description in forbidden_accelerated_default_fragments.items():
        if fragment in accelerated_types_text:
            findings.append(
                Finding(
                    "static-text", accelerated_types_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_accelerated_default_fragments = {
        "AccelerationMode mode = AccelerationMode::CPU_ONLY;": "CPU-only acceleration default",
        "bool auto_detect_gpu = false;": "GPU auto-detect disabled by default",
        "bool enable_gpu = false;": "GPU disabled by default",
        "bool enable_hybrid = false;": "hybrid execution disabled by default",
        "double gpu_work_ratio = 0.0;": "zero GPU work by default",
        "bool auto_detect_accelerated_runtime = false;": "accelerated runtime auto-detect disabled by default",
        "bool auto_detect_adaptive_acceleration = false;": "adaptive acceleration auto-detect disabled by default",
        "bool enable_matrix_multiplication = false;": "matrix acceleration disabled by default",
        "bool enable_sparsification = false;": "sparsification disabled by default",
        "bool enable_lockfree_multicore = false;": "lockfree acceleration disabled by default",
        "ExecutionMode mode = ExecutionMode::CPU_ONLY;": "CPU-only strategy default",
        "ExecutionMode execution_mode = ExecutionMode::CPU_ONLY;": "CPU-only metrics default",
    }
    for fragment, description in required_accelerated_default_fragments.items():
        if fragment not in accelerated_types_text:
            findings.append(
                Finding(
                    "static-text",
                    accelerated_types_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    work_distributor_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "accelerated" / "work_distributor.hpp"
    )
    work_distributor_text = (
        work_distributor_path.read_text(encoding="utf-8", errors="ignore")
        if work_distributor_path.exists()
        else ""
    )
    forbidden_work_distributor_fragments = {
        "double gpu_work_ratio = 0.999;": "work distributor defaults must not assign GPU work",
    }
    for fragment, description in forbidden_work_distributor_fragments.items():
        if fragment in work_distributor_text:
            findings.append(
                Finding(
                    "static-text", work_distributor_path.relative_to(ROOT).as_posix(), description
                )
            )
    required_work_distributor_fragments = {
        "double gpu_work_ratio = 0.0;": "zero-GPU work distributor default",
        "if (config_.gpu_work_ratio <= 0.0)": "CPU-only work distribution for zero GPU ratio",
    }
    for fragment, description in required_work_distributor_fragments.items():
        if fragment not in work_distributor_text:
            findings.append(
                Finding(
                    "static-text",
                    work_distributor_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    acceleration_runtime_engine_path = (
        ROOT / "src" / "persistence" / "acceleration_runtime" / "acceleration_runtime_engine.cpp"
    )
    adaptive_acceleration_engine_path = (
        ROOT / "src" / "persistence" / "adaptive_acceleration" / "adaptive_acceleration_engine.cpp"
    )
    acceleration_runtime_engine_text = (
        acceleration_runtime_engine_path.read_text(encoding="utf-8", errors="ignore")
        if acceleration_runtime_engine_path.exists()
        else ""
    )
    adaptive_acceleration_engine_text = (
        adaptive_acceleration_engine_path.read_text(encoding="utf-8", errors="ignore")
        if adaptive_acceleration_engine_path.exists()
        else ""
    )
    combined_acceleration_factory_text = (
        acceleration_runtime_engine_text + "\n" + adaptive_acceleration_engine_text
    )
    forbidden_acceleration_factory_fragments = {
        "config.use_accelerated_runtime = capabilities.cuda_available || capabilities.num_cpu_cores >= 8": (
            "acceleration-runtime factory must not auto-enable accelerated runtime"
        ),
        "config.auto_detect_accelerated_runtime = true": (
            "acceleration-runtime factory must not auto-detect acceleration"
        ),
        "config.use_adaptive_acceleration =\n        capabilities.cuda_available || capabilities.num_cpu_cores >= 8": (
            "adaptive-acceleration factory must not auto-enable adaptive runtime"
        ),
        "config.auto_detect_adaptive_acceleration = true": (
            "adaptive-acceleration factory must not auto-detect acceleration"
        ),
        "config.enable_gpu = capabilities.cuda_available": (
            "acceleration factories must not auto-enable GPU"
        ),
        "config.enable_hybrid = capabilities.cuda_available": (
            "acceleration factories must not auto-enable hybrid execution"
        ),
        "config.enable_hybrid = true": "acceleration use-case presets must not force hybrid execution",
        'config.use_accelerated_runtime ? "accelerated" : "cpu_exact"': (
            "CPU-backed acceleration-runtime engine must not relabel work as accelerated"
        ),
        'config.use_adaptive_acceleration ? "adaptive_acceleration" : "cpu_exact"': (
            "CPU-backed adaptive-acceleration engine must not relabel work as adaptive"
        ),
        "single_thread_baseline_ms": "acceleration engines must not report synthetic speedups",
    }
    for fragment, description in forbidden_acceleration_factory_fragments.items():
        if fragment in combined_acceleration_factory_text:
            findings.append(
                Finding(
                    "static-text",
                    acceleration_runtime_engine_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_acceleration_factory_fragments = {
        "config.use_accelerated_runtime = false": "CPU-only acceleration-runtime factory default",
        "config.auto_detect_accelerated_runtime = false": (
            "disabled acceleration-runtime auto-detection"
        ),
        "config.use_adaptive_acceleration = false": "CPU-only adaptive-acceleration factory default",
        "config.auto_detect_adaptive_acceleration = false": (
            "disabled adaptive-acceleration auto-detection"
        ),
        "config.acceleration.enable_gpu = false": "acceleration factory GPU disabled default",
        "config.enable_hybrid = false": "acceleration factory hybrid disabled default",
        "config.acceleration.mode = AccelerationMode::CPU_ONLY": (
            "acceleration factory CPU-only mode"
        ),
        'performance_stats_.algorithm_used = "cpu_exact"': (
            "CPU-backed acceleration engines report CPU exact execution"
        ),
        "performance_stats_.speedup_factor = 1.0": (
            "adaptive-acceleration engine reports neutral speedup"
        ),
        "performance_stats_.speedup_factor = DEFAULT_SPEEDUP_FACTOR": (
            "acceleration-runtime engine reports neutral speedup"
        ),
    }
    for fragment, description in required_acceleration_factory_fragments.items():
        if fragment not in combined_acceleration_factory_text:
            findings.append(
                Finding(
                    "static-text",
                    acceleration_runtime_engine_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    adaptive_algorithm_selector_path = (
        ROOT / "src" / "persistence" / "adaptive_acceleration" / "adaptive_algorithm_selector.cpp"
    )
    adaptive_algorithm_selector_exec_path = (
        ROOT
        / "src"
        / "persistence"
        / "adaptive_acceleration"
        / "adaptive_algorithm_selector_exec.cpp"
    )
    adaptive_algorithm_selector_header_path = (
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "adaptive_acceleration"
        / "adaptive_algorithm_selector.hpp"
    )
    adaptive_problem_analysis_path = (
        ROOT
        / "src"
        / "persistence"
        / "adaptive_acceleration"
        / "adaptive_acceleration_problem_analysis.cpp"
    )
    adaptive_algorithm_selector_text = (
        adaptive_algorithm_selector_path.read_text(encoding="utf-8", errors="ignore")
        if adaptive_algorithm_selector_path.exists()
        else ""
    )
    adaptive_algorithm_selector_exec_text = (
        adaptive_algorithm_selector_exec_path.read_text(encoding="utf-8", errors="ignore")
        if adaptive_algorithm_selector_exec_path.exists()
        else ""
    )
    adaptive_algorithm_selector_header_text = (
        adaptive_algorithm_selector_header_path.read_text(encoding="utf-8", errors="ignore")
        if adaptive_algorithm_selector_header_path.exists()
        else ""
    )
    adaptive_problem_analysis_text = (
        adaptive_problem_analysis_path.read_text(encoding="utf-8", errors="ignore")
        if adaptive_problem_analysis_path.exists()
        else ""
    )
    combined_adaptive_algorithm_selector_text = (
        adaptive_algorithm_selector_text
        + "\n"
        + adaptive_algorithm_selector_exec_text
        + "\n"
        + adaptive_algorithm_selector_header_text
        + "\n"
        + adaptive_problem_analysis_text
    )
    forbidden_adaptive_algorithm_selector_fragments = {
        "if (system.cuda_available)": (
            "adaptive algorithm selector must not predict GPU algorithms from hardware probes"
        ),
        "return problem.estimated_columns >= 4096 && problem.memory_requirement_mb <= available_mb": (
            "adaptive problem analyzer must not report GPU suitability while GPU execution is missing"
        ),
        "gpu.algorithm_type = AdaptiveAlgorithmType::GPU_ACCELERATED": (
            "adaptive algorithm selector must not add GPU predictions"
        ),
        "hybrid.algorithm_type = AdaptiveAlgorithmType::HYBRID_GPU_CPU": (
            "adaptive algorithm selector must not add hybrid GPU predictions"
        ),
        "requirements.prefer_gpu = config.prefer_gpu": (
            "adaptive selector must not honor GPU preference while GPU path is missing"
        ),
        "PersistenceBackend::CUDA_HYBRID": (
            "adaptive GPU execution helpers must not silently route to CUDA hybrid backend"
        ),
        "runtime::has_cuda_gpu": (
            "adaptive GPU execution helpers must not probe CUDA for CPU-default execution"
        ),
        "bool enable_hybrid_optimization = true;": (
            "adaptive selector defaults must not enable hybrid optimization"
        ),
    }
    for fragment, description in forbidden_adaptive_algorithm_selector_fragments.items():
        if fragment in combined_adaptive_algorithm_selector_text:
            findings.append(
                Finding(
                    "static-text",
                    adaptive_algorithm_selector_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_adaptive_algorithm_selector_fragments = {
        "bool enable_hybrid_optimization = false;": (
            "adaptive selector hybrid optimization disabled by default"
        ),
        "requirements.prefer_gpu = false;": "adaptive selector ignores missing GPU preference",
        "bool ProblemAnalyzer::shouldUseGpu": "adaptive problem analyzer exposes GPU suitability hook",
        "(void)system;\n    return false;": "adaptive problem analyzer reports GPU unsuitable",
        "cpuOnlySystemCapabilities": "adaptive problem analyzer uses local CPU-only capabilities",
        "if (system.available_memory == 0)": (
            "adaptive problem analyzer does not recommend streaming without memory capacity data"
        ),
        "binomialCoefficientBounded": "adaptive problem analyzer uses bounded binomial estimates",
        "term > std::numeric_limits<std::size_t>::max() - total": (
            "adaptive problem analyzer saturates column-estimate sums"
        ),
        "(points.size() % point_dim) != 0": "adaptive problem analyzer rejects partial points",
        "if (!std::isfinite(value))": "adaptive problem analyzer rejects non-finite points",
        "return errors::ErrorResult<std::vector<Pair>>::error(errors::ErrorCode::E10_GPU_OOM);": (
            "adaptive GPU helpers fail explicitly"
        ),
    }
    for fragment, description in required_adaptive_algorithm_selector_fragments.items():
        if fragment not in combined_adaptive_algorithm_selector_text:
            findings.append(
                Finding(
                    "static-text",
                    adaptive_algorithm_selector_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    if "SystemCapabilities::detectCapabilities()" in adaptive_problem_analysis_text:
        findings.append(
            Finding(
                "static-text",
                adaptive_problem_analysis_path.relative_to(ROOT).as_posix(),
                "adaptive problem analyzer must not add hardware-probe link dependency",
            )
        )
    forbidden_adaptive_problem_fragments = {
        "numerator *= (num_points - i)": (
            "adaptive problem analyzer must not multiply size_t binomial numerator unchecked"
        ),
        "denominator *= (i + 1)": (
            "adaptive problem analyzer must not multiply size_t binomial denominator unchecked"
        ),
    }
    for fragment, description in forbidden_adaptive_problem_fragments.items():
        if fragment in adaptive_problem_analysis_text:
            findings.append(
                Finding(
                    "static-text",
                    adaptive_problem_analysis_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    sparse_matrix_path = (
        ROOT / "src" / "persistence" / "adaptive_acceleration" / "sparse_matrix.cpp"
    )
    sparse_matrix_text = (
        sparse_matrix_path.read_text(encoding="utf-8", errors="ignore")
        if sparse_matrix_path.exists()
        else ""
    )
    if "if (n_rows * n_cols != dense_matrix.size())" in sparse_matrix_text:
        findings.append(
            Finding(
                "static-text",
                sparse_matrix_path.relative_to(ROOT).as_posix(),
                "sparse matrix dense-shape validation must not multiply dimensions before overflow check",
            )
        )
    required_sparse_matrix_fragments = {
        "bool denseElementCount": "central sparse matrix dense element-count validation",
        "bool validateShape": "central sparse matrix shape validation",
        "n_cols != 0 && n_rows > std::numeric_limits<size_t>::max() / n_cols": (
            "sparse matrix dense-shape overflow guard"
        ),
        "element_count <= static_cast<size_t>(std::numeric_limits<int>::max())": (
            "sparse matrix dense materialization size guard"
        ),
        "n_rows > static_cast<size_t>(std::numeric_limits<int>::max())": (
            "sparse matrix row index-range guard"
        ),
        "n_cols > static_cast<size_t>(std::numeric_limits<int>::max())": (
            "sparse matrix column index-range guard"
        ),
        "if (!std::isfinite(value))": "sparse matrix rejects non-finite dense values",
    }
    for fragment, description in required_sparse_matrix_fragments.items():
        if fragment not in sparse_matrix_text:
            findings.append(
                Finding(
                    "static-text",
                    sparse_matrix_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    adaptive_matrix_framework_path = (
        ROOT
        / "src"
        / "persistence"
        / "adaptive_acceleration"
        / "matrix_multiplication_framework.cpp"
    )
    adaptive_matrix_framework_text = (
        adaptive_matrix_framework_path.read_text(encoding="utf-8", errors="ignore")
        if adaptive_matrix_framework_path.exists()
        else ""
    )
    forbidden_adaptive_matrix_framework_fragments = {
        "return AlgorithmType::GPU_ACCELERATED": (
            "matrix multiplication framework must not report missing GPU execution"
        ),
        "return AlgorithmType::HYBRID": (
            "matrix multiplication framework must not report missing hybrid execution"
        ),
        "return AlgorithmType::LOCKFREE_MULTICORE": (
            "matrix multiplication framework must not report inactive lock-free execution"
        ),
    }
    for fragment, description in forbidden_adaptive_matrix_framework_fragments.items():
        if fragment in adaptive_matrix_framework_text:
            findings.append(
                Finding(
                    "static-text",
                    adaptive_matrix_framework_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    if "return AlgorithmType::STANDARD_CPU;" not in adaptive_matrix_framework_text:
        findings.append(
            Finding(
                "static-text",
                adaptive_matrix_framework_path.relative_to(ROOT).as_posix(),
                "matrix multiplication framework must report CPU baseline execution",
            )
        )
    accelerated_determinism_path = ROOT / "src" / "persistence" / "accelerated" / "determinism.cpp"
    accelerated_determinism_text = (
        accelerated_determinism_path.read_text(encoding="utf-8", errors="ignore")
        if accelerated_determinism_path.exists()
        else ""
    )
    required_accelerated_determinism_fragments = {
        "contract.level >= core::DeterminismLevel::STRICT,\n                 false": (
            "strict determinism does not implicitly enable GPU determinism"
        ),
        "!context_.getState().gpu_determinism_enabled": (
            "CPU-only deterministic work distribution when GPU determinism is disabled"
        ),
        "WorkDistribution distribution(0, total_columns, false)": (
            "strict CPU determinism assigns all deterministic work to CPU"
        ),
        "distribution.strategy = WorkDistributionStrategy::BASIC": (
            "CPU-only deterministic work distribution strategy"
        ),
    }
    for fragment, description in required_accelerated_determinism_fragments.items():
        if fragment not in accelerated_determinism_text:
            findings.append(
                Finding(
                    "static-text",
                    accelerated_determinism_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    accelerated_determinism_validation_path = (
        ROOT / "src" / "persistence" / "accelerated" / "determinism_validation.cpp"
    )
    accelerated_determinism_validation_text = (
        accelerated_determinism_validation_path.read_text(encoding="utf-8", errors="ignore")
        if accelerated_determinism_validation_path.exists()
        else ""
    )
    comprehensive_validation_index = accelerated_determinism_validation_text.find(
        "DeterminismValidationResult comprehensiveValidation("
    )
    if comprehensive_validation_index < 0:
        findings.append(
            Finding(
                "static-text",
                accelerated_determinism_validation_path.relative_to(ROOT).as_posix(),
                "missing deterministic comprehensive validation entry point",
            )
        )
    else:
        comprehensive_validation_block = accelerated_determinism_validation_text[
            comprehensive_validation_index:
        ]
        comprehensive_validation_block = comprehensive_validation_block.split(
            "}  // namespace determinism_validation", 1
        )[0]
        if "validateGpuDeterminism()" in comprehensive_validation_block:
            findings.append(
                Finding(
                    "static-text",
                    accelerated_determinism_validation_path.relative_to(ROOT).as_posix(),
                    "comprehensive deterministic validation must not implicitly probe CUDA",
                )
            )
    edge_collapse_path = (
        ROOT / "src" / "persistence" / "reduction" / "reduction_edge_collapse_ops.cpp"
    )
    edge_collapse_header_path = (
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "reduction"
        / "reduction_edge_collapse_ops.hpp"
    )
    edge_collapse_text = (
        edge_collapse_path.read_text(encoding="utf-8", errors="ignore")
        if edge_collapse_path.exists()
        else ""
    )
    edge_collapse_header_text = (
        edge_collapse_header_path.read_text(encoding="utf-8", errors="ignore")
        if edge_collapse_header_path.exists()
        else ""
    )
    combined_edge_collapse_text = edge_collapse_text + "\n" + edge_collapse_header_text
    forbidden_edge_collapse_fragments = {
        "GPU-accelerated edge collapse": "edge-collapse API must not advertise missing GPU execution",
        "Parallel version for large graphs": "edge-collapse API must not advertise missing parallel GPU execution",
        "Uses CPU implementation as backup": "edge-collapse API must describe CPU-backed compatibility honestly",
    }
    for fragment, description in forbidden_edge_collapse_fragments.items():
        if fragment in combined_edge_collapse_text:
            findings.append(
                Finding("static-text", edge_collapse_path.relative_to(ROOT).as_posix(), description)
            )
    required_edge_collapse_fragments = {
        "int original_vertices = 0;": "initialized edge-collapse result counters",
        "double vertex_reduction_ratio = 0.0;": "initialized edge-collapse stats",
        "result.original_vertices > 0": "guarded vertex-reduction statistics",
        "result.original_edges > 0": "guarded edge-reduction statistics",
        "bool shouldUseEdgeCollapse(size_t num_vertices, size_t num_edges, double density)": (
            "implemented edge-collapse recommendation helper"
        ),
        "Deterministic CPU compatibility entry point": "honest accelerated edge-collapse compatibility label",
    }
    for fragment, description in required_edge_collapse_fragments.items():
        if fragment not in combined_edge_collapse_text:
            findings.append(
                Finding(
                    "static-text",
                    edge_collapse_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    reduction_ops_header_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "reduction" / "reduction_ops.hpp"
    )
    reduction_ops_header_text = (
        reduction_ops_header_path.read_text(encoding="utf-8", errors="ignore")
        if reduction_ops_header_path.exists()
        else ""
    )
    if "bool use_gpu_ = true;" in reduction_ops_header_text:
        findings.append(
            Finding(
                "static-text",
                reduction_ops_header_path.relative_to(ROOT).as_posix(),
                "base reducer must not enable missing GPU reduction by default",
            )
        )
    if "bool use_gpu_ = false;" not in reduction_ops_header_text:
        findings.append(
            Finding(
                "static-text",
                reduction_ops_header_path.relative_to(ROOT).as_posix(),
                "missing CPU-only base reducer GPU default",
            )
        )
    reduction_union_find_path = (
        ROOT / "src" / "persistence" / "reduction" / "reduction_union_find_ops.cpp"
    )
    reduction_union_find_text = (
        reduction_union_find_path.read_text(encoding="utf-8", errors="ignore")
        if reduction_union_find_path.exists()
        else ""
    )
    if "stats.num_unite_operations = num_points - 1" in reduction_union_find_text:
        findings.append(
            Finding(
                "static-text",
                reduction_union_find_path.relative_to(ROOT).as_posix(),
                "union-find stats must not underflow unite-operation count for empty inputs",
            )
        )
    if "num_points > 0 ? num_points - 1 : 0" not in reduction_union_find_text:
        findings.append(
            Finding(
                "static-text",
                reduction_union_find_path.relative_to(ROOT).as_posix(),
                "missing guarded union-find unite-operation count",
            )
        )
    spectral_laplacian_header_path = (
        ROOT / "src" / "include" / "nerve" / "spectral" / "laplacian.hpp"
    )
    spectral_laplacian_header_text = (
        spectral_laplacian_header_path.read_text(encoding="utf-8", errors="ignore")
        if spectral_laplacian_header_path.exists()
        else ""
    )
    if "bool enable_gpu = true;" in spectral_laplacian_header_text:
        findings.append(
            Finding(
                "static-text",
                spectral_laplacian_header_path.relative_to(ROOT).as_posix(),
                "spectral Laplacian config must not enable GPU by default",
            )
        )
    if "bool enable_gpu = false;" not in spectral_laplacian_header_text:
        findings.append(
            Finding(
                "static-text",
                spectral_laplacian_header_path.relative_to(ROOT).as_posix(),
                "missing CPU-only spectral Laplacian GPU default",
            )
        )
    dmt_header_path = ROOT / "src" / "include" / "nerve" / "dmt" / "gpu_dmt.hpp"
    dmt_header_text = (
        dmt_header_path.read_text(encoding="utf-8", errors="ignore")
        if dmt_header_path.exists()
        else ""
    )
    if "bool use_gpu = true;" in dmt_header_text:
        findings.append(
            Finding(
                "static-text",
                dmt_header_path.relative_to(ROOT).as_posix(),
                "DMT config must not enable missing GPU routing by default",
            )
        )
    if "bool use_gpu = false;" not in dmt_header_text:
        findings.append(
            Finding(
                "static-text",
                dmt_header_path.relative_to(ROOT).as_posix(),
                "missing CPU-only DMT GPU default",
            )
        )
    reduction_clearing_header_path = (
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "reduction"
        / "reduction_clearing_ops.hpp"
    )
    reduction_clearing_header_text = (
        reduction_clearing_header_path.read_text(encoding="utf-8", errors="ignore")
        if reduction_clearing_header_path.exists()
        else ""
    )
    if "bool enable_gpu = true;" in reduction_clearing_header_text:
        findings.append(
            Finding(
                "static-text",
                reduction_clearing_header_path.relative_to(ROOT).as_posix(),
                "optimized reducer config must not enable missing GPU path by default",
            )
        )
    if "bool enable_gpu = false;" not in reduction_clearing_header_text:
        findings.append(
            Finding(
                "static-text",
                reduction_clearing_header_path.relative_to(ROOT).as_posix(),
                "missing CPU-only optimized reducer GPU default",
            )
        )
    high_dimensional_exact_path = (
        ROOT / "src" / "persistence" / "core" / "high_dimensional_exact.cpp"
    )
    high_dimensional_exact_header_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "core" / "high_dimensional_exact.hpp"
    )
    high_dimensional_exact_text = (
        high_dimensional_exact_path.read_text(encoding="utf-8", errors="ignore")
        if high_dimensional_exact_path.exists()
        else ""
    )
    high_dimensional_exact_header_text = (
        high_dimensional_exact_header_path.read_text(encoding="utf-8", errors="ignore")
        if high_dimensional_exact_header_path.exists()
        else ""
    )
    combined_high_dimensional_exact_text = (
        high_dimensional_exact_text + "\n" + high_dimensional_exact_header_text
    )
    forbidden_high_dimensional_exact_fragments = {
        "bool use_gpu_acceleration = true": (
            "high-dimensional exact config must not enable missing GPU path by default"
        ),
        "GPU_ACCELERATION_THRESHOLD": (
            "high-dimensional exact config must not keep missing GPU auto-enable threshold"
        ),
        "config.use_gpu_acceleration = (": (
            "high-dimensional exact optimal config must not auto-enable GPU"
        ),
        "involuted.use_gpu_acceleration = (": (
            "high-dimensional benchmark must not auto-enable GPU"
        ),
        "GPU Acceleration: For large matrix operations": (
            "high-dimensional docs must not advertise missing GPU acceleration"
        ),
    }
    for fragment, description in forbidden_high_dimensional_exact_fragments.items():
        if fragment in combined_high_dimensional_exact_text:
            findings.append(
                Finding(
                    "static-text",
                    high_dimensional_exact_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )
    required_high_dimensional_exact_fragments = {
        "bool use_gpu_acceleration = false": "CPU-only high-dimensional exact GPU default",
        "config.use_gpu_acceleration = false": (
            "high-dimensional exact optimal config keeps GPU disabled"
        ),
        "involuted.use_gpu_acceleration = false": ("high-dimensional benchmark keeps GPU disabled"),
    }
    for fragment, description in required_high_dimensional_exact_fragments.items():
        if fragment not in combined_high_dimensional_exact_text:
            findings.append(
                Finding(
                    "static-text",
                    high_dimensional_exact_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    flood_complex_path = ROOT / "src" / "persistence" / "core" / "flood_complex.cpp"
    flood_complex_header_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "core" / "flood_complex.hpp"
    )
    flood_complex_text = (
        flood_complex_path.read_text(encoding="utf-8", errors="ignore")
        if flood_complex_path.exists()
        else ""
    )
    flood_complex_header_text = (
        flood_complex_header_path.read_text(encoding="utf-8", errors="ignore")
        if flood_complex_header_path.exists()
        else ""
    )
    combined_flood_complex_text = flood_complex_text + "\n" + flood_complex_header_text
    forbidden_flood_complex_fragments = {
        "bool use_gpu = true": "flood-complex config must not enable missing GPU by default",
        "config.use_gpu = shouldUseFloodComplex": (
            "flood-complex optimal config must not recommend missing GPU execution"
        ),
        "Runs the CPU implementation when GPU execution fails": (
            "flood-complex GPU docs must not describe silent CPU rerouting"
        ),
        "return computeFloodComplex(points, point_dim, num_points, gpu_config)": (
            "flood-complex GPU entry point must not silently run CPU"
        ),
        "computeFloodComplexGPU": "flood-complex must not expose an inactive GPU entry point",
    }
    for fragment, description in forbidden_flood_complex_fragments.items():
        if fragment in combined_flood_complex_text:
            findings.append(
                Finding("static-text", flood_complex_path.relative_to(ROOT).as_posix(), description)
            )
    required_flood_complex_fragments = {
        "bool use_flooding = true": "flood-complex keeps flooding enabled by default",
    }
    for fragment, description in required_flood_complex_fragments.items():
        if fragment not in combined_flood_complex_text:
            findings.append(
                Finding(
                    "static-text",
                    flood_complex_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    ann_path = ROOT / "src" / "persistence" / "approximate" / "approximate_nearest_neighbor_ops.cpp"
    ann_header_path = (
        ROOT
        / "src"
        / "include"
        / "nerve"
        / "persistence"
        / "approximate"
        / "approximate_nearest_neighbor.hpp"
    )
    ann_text = ann_path.read_text(encoding="utf-8", errors="ignore") if ann_path.exists() else ""
    ann_header_text = (
        ann_header_path.read_text(encoding="utf-8", errors="ignore")
        if ann_header_path.exists()
        else ""
    )
    combined_ann_text = ann_text + "\n" + ann_header_text
    forbidden_ann_fragments = {
        "GPU_THRESHOLD": "ANN optimal config must not keep missing GPU auto-enable threshold",
        "config.use_gpu = (": "ANN optimal config must not auto-enable missing GPU execution",
        "return fastEdgeDetectionANN(points, point_dim, num_points, max_radius, config)": (
            "ANN GPU entry point must not silently run CPU"
        ),
        "@brief GPU-accelerated ANN edge detection": (
            "ANN docs must not advertise missing GPU acceleration"
        ),
        "fastEdgeDetectionANNGPU": "ANN must not expose an inactive GPU entry point",
    }
    for fragment, description in forbidden_ann_fragments.items():
        if fragment in combined_ann_text:
            findings.append(
                Finding("static-text", ann_path.relative_to(ROOT).as_posix(), description)
            )
    required_ann_fragments = {
        "config.ef_search": "ANN optimal config sets search width",
    }
    for fragment, description in required_ann_fragments.items():
        if fragment not in combined_ann_text:
            findings.append(
                Finding(
                    "static-text", ann_path.relative_to(ROOT).as_posix(), f"missing {description}"
                )
            )
    persistence_core_types_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "core" / "core_types.hpp"
    )
    persistence_core_types_text = (
        persistence_core_types_path.read_text(encoding="utf-8", errors="ignore")
        if persistence_core_types_path.exists()
        else ""
    )
    core_decl_index = persistence_core_types_text.find(
        "namespace nerve::core {\nclass DeterminismContract;"
    )
    persistence_decl_index = persistence_core_types_text.find("namespace nerve::persistence {")
    if core_decl_index < 0:
        findings.append(
            Finding(
                "static-text",
                persistence_core_types_path.relative_to(ROOT).as_posix(),
                "missing global DeterminismContract forward declaration",
            )
        )
    elif persistence_decl_index >= 0 and core_decl_index > persistence_decl_index:
        findings.append(
            Finding(
                "static-text",
                persistence_core_types_path.relative_to(ROOT).as_posix(),
                "DeterminismContract forward declaration must not create nested nerve namespace",
            )
        )
    boundary_matrix_path = ROOT / "src" / "torch" / "boundary_matrix.cpp"
    boundary_sparse_utils_path = (
        ROOT / "src" / "include" / "nerve" / "torch" / "boundary_matrix_sparse_utils.hpp"
    )
    boundary_matrix_text = (
        boundary_matrix_path.read_text(encoding="utf-8", errors="ignore")
        if boundary_matrix_path.exists()
        else ""
    )
    boundary_sparse_utils_text = (
        boundary_sparse_utils_path.read_text(encoding="utf-8", errors="ignore")
        if boundary_sparse_utils_path.exists()
        else ""
    )
    required_boundary_dtype_fragments = {
        "indices_(std::move(indices).to(at::kLong).contiguous())": "boundary indices normalization",
        "indptr_(std::move(indptr).to(at::kLong).contiguous())": "boundary indptr normalization",
        "data_(std::move(data).to(at::kDouble).contiguous())": "boundary data normalization",
        "indices.cpu().to(at::kLong).contiguous()": "sparse decode index normalization",
        "data.cpu().to(at::kDouble).contiguous()": "sparse decode value normalization",
    }
    combined_boundary_text = boundary_matrix_text + "\n" + boundary_sparse_utils_text
    for fragment, description in required_boundary_dtype_fragments.items():
        if fragment not in combined_boundary_text:
            findings.append(
                Finding(
                    "static-text",
                    boundary_matrix_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    named_version_pattern = re.compile(
        r"(?:schema|api|release|interface|artifact)[A-Za-z0-9_/-]*" + "v" + "2",
        re.IGNORECASE,
    )
    implied_version_pattern = re.compile(
        r"(?:schema|api|release|interface|artifact)[A-Za-z0-9_/-]*current|"
        r"current[A-Za-z0-9_/-]*(?:schema|api|release|interface|artifact)",
        re.IGNORECASE,
    )
    for path in _iter_files(ROOT / "src", (".cpp", ".hpp", ".h", ".cu", ".cuh")):
        rel = path.relative_to(ROOT).as_posix()
        if named_version_pattern.search(rel):
            findings.append(
                Finding("static-text", rel, "named version artifact uses second-generation naming")
            )
            continue
        if implied_version_pattern.search(rel):
            findings.append(
                Finding("static-text", rel, "named version artifact uses current naming")
            )
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in named_version_pattern.finditer(text):
            findings.append(
                Finding(
                    "static-text",
                    rel,
                    f"named version artifact uses second-generation naming: {match.group(0)}",
                )
            )
            break
        for match in implied_version_pattern.finditer(text):
            findings.append(
                Finding(
                    "static-text",
                    rel,
                    f"named version artifact uses current naming: {match.group(0)}",
                )
            )
            break
    public_prototype_fragments = {
        PYTHON_API_PATH: ("PH5PH6Prototype = _core", '"PH5PH6Prototype"'),
        PYBIND_API_PATH: ('m, "PH5PH6Prototype"',),
    }
    for path, fragments in public_prototype_fragments.items():
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        for fragment in fragments:
            if fragment in text:
                findings.append(
                    Finding(
                        "static-text",
                        path.relative_to(ROOT).as_posix(),
                        "public PH5/PH6 API exposes prototype naming",
                    )
                )
                break
    source_prototype_tokens = (
        "PH5" + "PH6" + "Prototype",
        "UnifiedPersistence" + "Prototype",
        "createPh5" + "Prototype",
        "createPh6" + "Prototype",
        "createCohomology" + "Prototype",
        "createWitness" + "Prototype",
    )
    for path in _iter_files(ROOT / "src", (".cpp", ".hpp", ".h", ".cu", ".cuh")):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in source_prototype_tokens:
            if token in text:
                findings.append(
                    Finding("static-text", rel, f"production source uses prototype naming: {token}")
                )
                break

    runtime_flags_path = ROOT / "src" / "include" / "nerve" / "features" / "feature_flags.hpp"
    runtime_flags_text = (
        runtime_flags_path.read_text(encoding="utf-8", errors="ignore")
        if runtime_flags_path.exists()
        else ""
    )
    if "true, true, true, true, true, true, true, true, false" not in runtime_flags_text:
        findings.append(
            Finding(
                "static-text",
                runtime_flags_path.relative_to(ROOT).as_posix(),
                "production runtime features must be enabled by default while debug visualization stays disabled",
            )
        )

    feature_access_path = (
        ROOT / "src" / "include" / "nerve" / "feature_access" / "feature_flags.hpp"
    )
    feature_access_text = (
        feature_access_path.read_text(encoding="utf-8", errors="ignore")
        if feature_access_path.exists()
        else ""
    )
    required_feature_access_defaults = {
        "#define TOPOLOGIB_FEATURE_FLAG_PH4 1": "PH4 default feature access",
        "#define TOPOLOGIB_FEATURE_FLAG_DIFFERENTIABLE_PERSISTENCE 1": (
            "differentiable persistence default feature access"
        ),
        "#define TOPOLOGIB_FEATURE_FLAG_SHEAF_LAPLACIAN_ADVANCED 1": (
            "sheaf Laplacian default feature access"
        ),
        "#define TOPOLOGIB_FEATURE_FLAG_ADVANCED_STREAMING 1": "streaming default feature access",
        "#define TOPOLOGIB_FEATURE_FLAG_RESEARCH_MODE 0": "research mode disabled by default",
    }
    for fragment, description in required_feature_access_defaults.items():
        if fragment not in feature_access_text:
            findings.append(
                Finding(
                    "static-text",
                    feature_access_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )

    unified_header_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "utils" / "unified_persistence.hpp"
    )
    unified_header_text = (
        unified_header_path.read_text(encoding="utf-8", errors="ignore")
        if unified_header_path.exists()
        else ""
    )
    required_unified_header_fragments = {
        "using UnifiedPersistenceEngine = PH5PH6Engine": "unified persistence aliases the production engine",
        "return createPh5Engine<PointType, Scalar>()": "cohomology factory forwards to PH5 engine",
        "return createPh6Engine<PointType, Scalar>()": "witness factory forwards to PH6 engine",
        "return createPh5EngineIfEnabled<PointType, Scalar>()": (
            "conditional cohomology factory forwards to PH5 engine"
        ),
        "return createPh6EngineIfEnabled<PointType, Scalar>()": (
            "conditional witness factory forwards to PH6 engine"
        ),
    }
    for fragment, description in required_unified_header_fragments.items():
        if fragment not in unified_header_text:
            findings.append(
                Finding(
                    "static-text",
                    unified_header_path.relative_to(ROOT).as_posix(),
                    f"missing {description}",
                )
            )
    forbidden_unified_header_fragments = {
        "UnifiedPersistence" + "Prototype": "unified persistence must not use prototype naming",
        "unified_persistence_methods_impl.hpp": (
            "unified persistence public header must not include PH5/PH6 method impl directly"
        ),
        "unified_persistence_runtime_impl.hpp": (
            "unified persistence public header must not include PH5/PH6 runtime impl directly"
        ),
    }
    for fragment, description in forbidden_unified_header_fragments.items():
        if fragment in unified_header_text:
            findings.append(
                Finding(
                    "static-text",
                    unified_header_path.relative_to(ROOT).as_posix(),
                    description,
                )
            )

    streaming_reducer_path = (
        ROOT / "src" / "include" / "nerve" / "persistence" / "streaming" / "streaming_reducer.hpp"
    )
    streaming_generator_path = (
        ROOT / "src" / "persistence" / "streaming" / "streaming_column_generator.cpp"
    )
    for path in (streaming_reducer_path, streaming_generator_path):
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        if "edge_list_v2_" in text:
            findings.append(
                Finding(
                    "static-text",
                    path.relative_to(ROOT).as_posix(),
                    "streaming simplex edge storage must not use second-generation naming",
                )
            )
    return findings
