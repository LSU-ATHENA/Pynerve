
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nerve::runtime
{

enum class ProbeStatus
{
    kOk,
    kMissing,
    kError,
};

std::string probe_status_to_string(ProbeStatus status);

template <typename T>
struct ProbeValue
{
    ProbeStatus status = ProbeStatus::kMissing;
    T value{};
    std::string diagnostics;

    bool ok() const { return status == ProbeStatus::kOk; }
};

struct CpuTopology
{
    std::size_t logical_cores = 0;
    std::size_t physical_cores = 0;
    std::string model;
};

struct NumaNodeInfo
{
    std::size_t node_id = 0;
    std::vector<std::size_t> cpu_ids;
    std::uint64_t total_memory_bytes = 0;
};

struct GpuDeviceInfo
{
    int device_id = -1;
    std::string name;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    std::uint64_t total_memory_bytes = 0;
    std::uint64_t free_memory_bytes = 0;
    bool supports_tensor_cores = false;
};

struct HardwareSnapshot
{
    std::uint64_t collected_unix_ms = 0;

    ProbeValue<CpuTopology> cpu;
    ProbeValue<std::uint64_t> total_memory_bytes;
    ProbeValue<std::uint64_t> available_memory_bytes;
    ProbeValue<std::vector<NumaNodeInfo>> numa_nodes;
    ProbeValue<std::vector<GpuDeviceInfo>> gpus;

    std::vector<std::string> diagnostics;
};

HardwareSnapshot collectHardwareSnapshot();

std::string getHardwareFingerprint(const HardwareSnapshot &snapshot);

bool has_cuda_gpu(const HardwareSnapshot &snapshot);

} // namespace nerve::runtime
