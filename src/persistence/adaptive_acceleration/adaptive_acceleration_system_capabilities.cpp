
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_system_capabilities.hpp"
#include "nerve/platform.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace nerve::persistence::adaptive_acceleration
{

namespace
{

double detectCpuClockGhz()
{
    std::ifstream stream("/proc/cpuinfo");
    if (!stream.is_open())
    {
        return 0.0;
    }
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.rfind("cpu MHz", 0) != 0)
        {
            continue;
        }
        const auto sep = line.find(':');
        if (sep == std::string::npos)
        {
            continue;
        }
        const std::string value = line.substr(sep + 1);
        std::stringstream parser(value);
        double mhz = 0.0;
        parser >> mhz;
        if (mhz > 0.0)
        {
            return mhz / 1000.0;
        }
    }
    return 0.0;
}

std::size_t estimateCacheSizeBytes(const runtime::HardwareSnapshot &snapshot, int level)
{
    if (snapshot.cpu.status != runtime::ProbeStatus::kOk)
    {
        return 0;
    }
    const std::size_t logical = std::max<std::size_t>(1, snapshot.cpu.value.logical_cores);
    switch (level)
    {
        case 1:
            return logical * 32 * 1024;
        case 2:
            return logical * 512 * 1024;
        case 3:
            return logical * 2 * 1024 * 1024;
        default:
            return 0;
    }
}

std::vector<CPUInfo> buildCpuTopology(const runtime::HardwareSnapshot &snapshot)
{
    std::vector<CPUInfo> topology;
    if (snapshot.cpu.status != runtime::ProbeStatus::kOk)
    {
        return topology;
    }

    std::unordered_map<std::size_t, std::size_t> cpu_to_node;
    if (snapshot.numa_nodes.status == runtime::ProbeStatus::kOk)
    {
        for (const auto &node : snapshot.numa_nodes.value)
        {
            for (std::size_t cpu_id : node.cpu_ids)
            {
                cpu_to_node[cpu_id] = node.node_id;
            }
        }
    }

    const std::size_t logical = std::max<std::size_t>(1, snapshot.cpu.value.logical_cores);
    topology.reserve(logical);
    for (std::size_t cpu_id = 0; cpu_id < logical; ++cpu_id)
    {
        CPUInfo info;
        info.cpu_id = cpu_id;
        info.core_id = cpu_id;
        info.node_id = cpu_to_node.count(cpu_id) ? cpu_to_node[cpu_id] : 0;
        info.is_performance_core = cpu_id < snapshot.cpu.value.physical_cores;
        info.cpu_model = snapshot.cpu.value.model;
        info.cache_info = "derived";
        topology.push_back(std::move(info));
    }
    return topology;
}

std::vector<std::size_t> buildNumaNodes(const runtime::HardwareSnapshot &snapshot)
{
    std::vector<std::size_t> nodes;
    if (snapshot.numa_nodes.status != runtime::ProbeStatus::kOk)
    {
        return nodes;
    }
    nodes.reserve(snapshot.numa_nodes.value.size());
    for (const auto &node : snapshot.numa_nodes.value)
    {
        nodes.push_back(node.node_id);
    }
    return nodes;
}

} // namespace

SystemCapabilities SystemCapabilities::detectCapabilities()
{
    SystemCapabilities caps;
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();

    if (snapshot.cpu.status == runtime::ProbeStatus::kOk)
    {
        caps.num_cpu_cores = std::max<std::size_t>(1, snapshot.cpu.value.logical_cores);
        caps.cpu_model = snapshot.cpu.value.model;
    }
    else
    {
        caps.num_cpu_cores = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        caps.cpu_model = "unknown";
    }

    if (snapshot.total_memory_bytes.status == runtime::ProbeStatus::kOk)
    {
        caps.total_memory = snapshot.total_memory_bytes.value;
    }
    if (snapshot.available_memory_bytes.status == runtime::ProbeStatus::kOk)
    {
        caps.available_memory = snapshot.available_memory_bytes.value;
    }

    caps.cpu_clock_ghz = detectCpuClockGhz();
    caps.l1_cache_size = estimateCacheSizeBytes(snapshot, 1);
    caps.l2_cache_size = estimateCacheSizeBytes(snapshot, 2);
    caps.l3_cache_size = estimateCacheSizeBytes(snapshot, 3);

    caps.cpu_topology = buildCpuTopology(snapshot);
    caps.numa_nodes = buildNumaNodes(snapshot);
    caps.has_numa_support = !caps.numa_nodes.empty();

    caps.cuda_available = runtime::has_cuda_gpu(snapshot);
    if (caps.cuda_available)
    {
        const auto &gpu = snapshot.gpus.value.front();
        caps.gpu_name = gpu.name;
        caps.compute_capability =
            (gpu.compute_capability_major * 10) + gpu.compute_capability_minor;
        caps.total_memory =
            std::max(caps.total_memory, static_cast<std::size_t>(gpu.total_memory_bytes));
        caps.available_memory =
            std::max(caps.available_memory, static_cast<std::size_t>(gpu.free_memory_bytes));

        caps.supports_fp16 = caps.compute_capability >= 60;
        caps.supports_tf32 = caps.compute_capability >= 80;
        caps.supports_bf16 = caps.compute_capability >= 80;
        caps.supports_tensor_cores = gpu.supports_tensor_cores;
        caps.supports_wmma = caps.compute_capability >= 70;
        caps.supports_mma = caps.compute_capability >= 80;
        caps.supports_fp64_tensor_cores = caps.compute_capability >= 90;
        caps.max_threads_per_block = 1024;
        caps.max_blocks_per_sm = std::max(1, caps.compute_capability / 10);
        caps.cuda_version = "runtime-probe";
        caps.driver_version = "runtime-probe";

        const double gpu_memory_gb =
            static_cast<double>(gpu.total_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
        caps.memory_bandwidth = std::max(50.0, gpu_memory_gb * 100.0);
    }
    else
    {
        caps.compute_capability = 0;
        caps.gpu_name.clear();
        caps.max_threads_per_block = 0;
        caps.max_blocks_per_sm = 0;
        caps.memory_bandwidth = std::max(1.0, static_cast<double>(caps.num_cpu_cores) * 12.0);
    }

    caps.theoretical_gflops =
        static_cast<double>(caps.num_cpu_cores) * std::max(0.5, caps.cpu_clock_ghz) * 16.0;
    if (caps.cuda_available)
    {
        caps.theoretical_gflops += static_cast<double>(caps.compute_capability) * 200.0;
    }

    return caps;
}

bool SystemCapabilities::is_cuda_available()
{
    return runtime::has_cuda_gpu(runtime::collectHardwareSnapshot());
}

bool SystemCapabilities::isArchitectureSupported(int compute_capability)
{
    return compute_capability >= getMinComputeCapability();
}

int SystemCapabilities::getMinComputeCapability()
{
    return 60;
}

std::size_t SystemCapabilities::getOptimalThreadCount()
{
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (snapshot.cpu.status == runtime::ProbeStatus::kOk && snapshot.cpu.value.logical_cores > 0)
    {
        return snapshot.cpu.value.logical_cores;
    }
    return std::max<std::size_t>(1, std::thread::hardware_concurrency());
}

NUMAConfig SystemCapabilities::getCurrentConfig()
{
    NUMAConfig config;
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    config.enable_numa = snapshot.numa_nodes.status == runtime::ProbeStatus::kOk &&
                         !snapshot.numa_nodes.value.empty();
    config.preferred_node = config.enable_numa ? snapshot.numa_nodes.value.front().node_id : 0;
    config.memory_interleaving = config.enable_numa && snapshot.numa_nodes.value.size() > 1;
    config.setThreadAffinity = true;
    config.round_robin_scheduling = config.enable_numa && snapshot.numa_nodes.value.size() > 1;
    config.cache_line_size = 64;
    return config;
}

NUMAConfig ThreadAffinityManager::current_config_;

void ThreadAffinityManager::setThreadAffinity(std::thread::id /*thread_id*/, std::size_t cpu_id)
{
    nerve::sys::CpuSet cpuset;
    cpuset.clear();
    cpuset.set(static_cast<int>(cpu_id));
    nerve::sys::thread_set_affinity(nerve::sys::thread_self(), &cpuset);
}

void ThreadAffinityManager::optimizeThreadPlacement(std::size_t num_threads,
                                                    const NUMAConfig &config)
{
    current_config_ = config;
    if (num_threads == 0 || !config.setThreadAffinity)
    {
        return;
    }
    const std::vector<CPUInfo> topology = getCpuTopology();
    if (topology.empty())
    {
        return;
    }
    setThreadAffinity(std::this_thread::get_id(), topology.front().cpu_id);
}

std::vector<std::size_t> ThreadAffinityManager::getNumaNodes()
{
    return buildNumaNodes(runtime::collectHardwareSnapshot());
}

std::vector<CPUInfo> ThreadAffinityManager::getCpuTopology()
{
    return buildCpuTopology(runtime::collectHardwareSnapshot());
}

} // namespace nerve::persistence::adaptive_acceleration
