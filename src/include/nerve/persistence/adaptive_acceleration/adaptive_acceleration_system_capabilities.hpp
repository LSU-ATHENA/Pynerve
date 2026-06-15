
#pragma once

#include <cstddef>
#include <string>
#include <thread>
#include <vector>

namespace nerve::persistence::adaptive_acceleration
{

struct GPUInfo
{
    int compute_capability = 0;
    std::string gpu_name;
    std::size_t total_memory = 0;
    std::size_t available_memory = 0;
    bool supports_fp16 = false;
    bool supports_tf32 = false;
    bool supports_bf16 = false;
    int max_threads_per_block = 0;
    int max_blocks_per_sm = 0;
    bool supports_tensor_cores = false;
    bool supports_wmma = false;
    bool supports_mma = false;
    bool supports_fp64_tensor_cores = false;
    std::string driver_version;
    std::string cuda_version;
};

struct CPUInfo
{
    std::size_t cpu_id = 0;
    std::size_t node_id = 0;
    std::size_t core_id = 0;
    std::vector<std::size_t> sibling_cores;
    bool is_performance_core = false;
    std::string cpu_model;
    std::string cache_info;
};

struct NUMAConfig
{
    bool enable_numa = true;
    std::size_t preferred_node = 0;
    bool memory_interleaving = true;
    bool setThreadAffinity = true;
    bool round_robin_scheduling = false;
    std::size_t cache_line_size = 64;
};

struct SystemCapabilities
{
    bool cuda_available = false;
    int compute_capability = 0;
    std::string gpu_name;
    std::size_t total_memory = 0;
    std::size_t available_memory = 0;
    bool supports_fp16 = false;
    bool supports_tf32 = false;
    bool supports_bf16 = false;
    int max_threads_per_block = 0;
    int max_blocks_per_sm = 0;

    std::size_t num_cpu_cores = 1;
    bool has_numa_support = false;
    std::vector<CPUInfo> cpu_topology;
    std::vector<std::size_t> numa_nodes;
    std::size_t l3_cache_size = 0;
    std::size_t l2_cache_size = 0;
    std::size_t l1_cache_size = 0;

    double theoretical_gflops = 0.0;
    double memory_bandwidth = 0.0;
    double cpu_clock_ghz = 0.0;

    bool supports_tensor_cores = false;
    bool supports_wmma = false;
    bool supports_mma = false;
    bool supports_fp64_tensor_cores = false;

    std::string cpu_model;
    std::string os_version;
    std::string driver_version;
    std::string cuda_version;

    static SystemCapabilities detectCapabilities();
    static bool is_cuda_available();
    static bool isArchitectureSupported(int compute_capability);
    static int getMinComputeCapability();
    static std::size_t getOptimalThreadCount();
    static NUMAConfig getCurrentConfig();
};

class ThreadAffinityManager
{
public:
    static void setThreadAffinity(std::thread::id thread_id, std::size_t cpu_id);
    static void optimizeThreadPlacement(std::size_t num_threads, const NUMAConfig &config);
    static std::vector<std::size_t> getNumaNodes();
    static std::vector<CPUInfo> getCpuTopology();

private:
    static NUMAConfig current_config_;
};

} // namespace nerve::persistence::adaptive_acceleration
