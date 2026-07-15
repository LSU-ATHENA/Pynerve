
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/adaptive_acceleration/adaptive_acceleration_system_capabilities.hpp"
#include "nerve/platform.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(__linux__)
#if __has_include(<numaif.h>)
#include <numaif.h>
#define NERVE_HAS_NUMA_POLICY 1
#else
#define NERVE_HAS_NUMA_POLICY 0
#endif
#else
#define NERVE_HAS_NUMA_POLICY 0
#endif

namespace nerve::persistence::adaptive_acceleration::gpu
{
namespace
{

struct ThreadPlacement
{
    std::vector<std::size_t> cpu_ids;
    runtime::ProbeStatus numa_status = runtime::ProbeStatus::kMissing;
    std::string diagnostics;
};

struct NumaPolicyResult
{
    runtime::ProbeStatus status = runtime::ProbeStatus::kMissing;
    std::string diagnostics;
};

std::vector<std::size_t> flattenNumaCpus(const std::vector<runtime::NumaNodeInfo> &nodes,
                                         std::size_t limit)
{
    std::vector<std::size_t> cpus;
    cpus.reserve(limit);
    if (nodes.empty() || limit == 0)
    {
        return cpus;
    }
    std::size_t node_index = 0;
    std::size_t cpu_offset = 0;
    while (cpus.size() < limit)
    {
        const runtime::NumaNodeInfo &node = nodes[node_index];
        if (cpu_offset < node.cpu_ids.size())
        {
            cpus.push_back(node.cpu_ids[cpu_offset]);
        }
        node_index = (node_index + 1) % nodes.size();
        if (node_index == 0)
        {
            ++cpu_offset;
            bool any_remaining = false;
            for (const runtime::NumaNodeInfo &candidate : nodes)
            {
                if (cpu_offset < candidate.cpu_ids.size())
                {
                    any_remaining = true;
                    break;
                }
            }
            if (!any_remaining)
            {
                break;
            }
        }
    }
    return cpus;
}

ThreadPlacement buildThreadPlacement(std::size_t requested_threads)
{
    ThreadPlacement placement;
    if (requested_threads == 0)
    {
        return placement;
    }

    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    placement.numa_status = snapshot.numa_nodes.status;
    placement.diagnostics = snapshot.numa_nodes.diagnostics;

    if (snapshot.numa_nodes.ok() && !snapshot.numa_nodes.value.empty())
    {
        placement.cpu_ids = flattenNumaCpus(snapshot.numa_nodes.value, requested_threads);
    }
    if (placement.cpu_ids.size() < requested_threads)
    {
        const std::size_t logical_cores =
            snapshot.cpu.ok() ? std::max<std::size_t>(1, snapshot.cpu.value.logical_cores)
                              : std::max<std::size_t>(1, requested_threads);
        for (std::size_t cpu = 0; placement.cpu_ids.size() < requested_threads; ++cpu)
        {
            placement.cpu_ids.push_back(cpu % logical_cores);
        }
    }
    return placement;
}

errors::ErrorResult<void> pinCurrentThreadToCpu(std::size_t cpu_id)
{
    nerve::sys::CpuSet set;
    set.clear();
    set.set(static_cast<int>(cpu_id));
    const int rc = nerve::sys::thread_set_affinity(nerve::sys::thread_self(), &set);
    if (rc != 0)
    {
        return errors::ErrorResult<void>::error(errors::ErrorCode::E61_NUMA_AFFINITY_FAIL);
    }
    return errors::ErrorResult<void>::ok();
}

NumaPolicyResult setNumaMemoryPolicy(std::size_t node_id)
{
    NumaPolicyResult result;
#if NERVE_HAS_NUMA_POLICY
    constexpr std::size_t kMaskBits = sizeof(unsigned long) * 8;
    if (node_id >= kMaskBits)
    {
        result.status = runtime::ProbeStatus::kError;
        result.diagnostics = "node id exceeds nodemask capacity";
        return result;
    }
    const unsigned long nodemask = (1UL << node_id);
    const int rc = set_mempolicy(MPOL_BIND, &nodemask, kMaskBits);
    if (rc != 0)
    {
        result.status = runtime::ProbeStatus::kError;
        result.diagnostics = "setMempolicy failed";
        return result;
    }
    result.status = runtime::ProbeStatus::kOk;
    result.diagnostics = "numa memory policy applied";
    return result;
#else
    constexpr std::size_t kMaxNodes = sizeof(unsigned long) * 8;
    if (node_id >= kMaxNodes)
    {
        result.status = runtime::ProbeStatus::kError;
        result.diagnostics = "node id exceeds nodemask capacity";
        return result;
    }
    result.status = runtime::ProbeStatus::kMissing;
    result.diagnostics = "numa policy API missing on this platform";
    return result;
#endif
}

double estimateTensorCoreGain(const runtime::HardwareSnapshot &snapshot)
{
    if (!snapshot.gpus.ok() || snapshot.gpus.value.empty())
    {
        return 1.0;
    }
    const auto best_gpu =
        std::max_element(snapshot.gpus.value.begin(), snapshot.gpus.value.end(),
                         [](const runtime::GpuDeviceInfo &lhs, const runtime::GpuDeviceInfo &rhs) {
                             return lhs.compute_capability_major < rhs.compute_capability_major;
                         });
    if (best_gpu == snapshot.gpus.value.end())
    {
        return 1.0;
    }
    const double capability_factor = static_cast<double>(best_gpu->compute_capability_major) / 10.0;
    const double tensor_factor = best_gpu->supports_tensor_cores ? 1.25 : 1.0;
    return std::max(1.0, capability_factor * tensor_factor);
}

} // namespace

errors::ErrorResult<std::vector<std::size_t>>
getOptimalThreadPlacement(std::size_t num_threads, const SystemCapabilities &system)
{
    const std::size_t requested_threads =
        num_threads == 0 ? std::max<std::size_t>(1, system.num_cpu_cores) : num_threads;
    ThreadPlacement placement = buildThreadPlacement(requested_threads);
    if (placement.cpu_ids.empty())
    {
        return errors::ErrorResult<std::vector<std::size_t>>::error(
            errors::ErrorCode::E41_RESOURCE_LIMIT);
    }
    return errors::ErrorResult<std::vector<std::size_t>>::success(std::move(placement.cpu_ids));
}

errors::ErrorResult<void> optimizeCurrentThreadAffinity(std::size_t thread_index)
{
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    const std::size_t logical_cores =
        snapshot.cpu.ok() ? std::max<std::size_t>(1, snapshot.cpu.value.logical_cores) : 1;
    const std::size_t cpu_id = thread_index % logical_cores;
    return pinCurrentThreadToCpu(cpu_id);
}

runtime::ProbeValue<double> getEstimatedTensorCoreGain()
{
    runtime::ProbeValue<double> gain;
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (!snapshot.gpus.ok())
    {
        gain.status = snapshot.gpus.status;
        gain.diagnostics = snapshot.gpus.diagnostics;
        return gain;
    }
    gain.status = runtime::ProbeStatus::kOk;
    gain.value = estimateTensorCoreGain(snapshot);
    gain.diagnostics = "estimated from probed gpu capabilities";
    return gain;
}

runtime::ProbeValue<std::string> applyNumaPolicyForThread(std::size_t thread_index)
{
    runtime::ProbeValue<std::string> out;
    const runtime::HardwareSnapshot snapshot = runtime::collectHardwareSnapshot();
    if (!snapshot.numa_nodes.ok() || snapshot.numa_nodes.value.empty())
    {
        out.status = snapshot.numa_nodes.status;
        out.diagnostics = snapshot.numa_nodes.diagnostics;
        return out;
    }
    const std::size_t node_index = thread_index % snapshot.numa_nodes.value.size();
    const std::size_t node_id = snapshot.numa_nodes.value[node_index].node_id;
    NumaPolicyResult result = setNumaMemoryPolicy(node_id);
    out.status = result.status;
    out.diagnostics = result.diagnostics;
    if (result.status == runtime::ProbeStatus::kOk)
    {
        out.value = "bound_to_node_" + std::to_string(node_id);
    }
    return out;
}

} // namespace nerve::persistence::adaptive_acceleration::gpu
