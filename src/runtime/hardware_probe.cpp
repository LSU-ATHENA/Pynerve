#include "nerve/config.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(NERVE_HAS_CUDA_RUNTIME) || defined(NERVE_HAS_CUDA)
#include <cuda_runtime_api.h>
#endif
namespace nerve::runtime
{
namespace
{
constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
std::uint64_t unixTimeMs()
{
    using namespace std::chrono;
    return static_cast<std::uint64_t>(
        duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}
std::string trimCopy(const std::string &value)
{
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0)
    {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0)
    {
        --end;
    }
    return value.substr(begin, end - begin);
}
std::uint64_t fnv1aHash(const std::string &text)
{
    std::uint64_t hash = kFnvOffsetBasis;
    for (char ch : text)
    {
        hash ^= static_cast<unsigned char>(ch);
        hash *= kFnvPrime;
    }
    return hash;
}
bool parseSizeT(std::string_view text, std::size_t *value)
{
    std::string trimmed = trimCopy(std::string(text));
    if (trimmed.empty())
        return false;
    std::size_t parsed = 0;
    for (const char ch : trimmed)
    {
        if (std::isdigit(static_cast<unsigned char>(ch)) == 0)
        {
            return false;
        }
        const std::size_t digit = static_cast<std::size_t>(ch - '0');
        if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10U)
        {
            return false;
        }
        parsed = parsed * 10U + digit;
    }
    *value = parsed;
    return true;
}
std::uint64_t kilobytesToBytes(std::uint64_t value_kb)
{
    return value_kb > std::numeric_limits<std::uint64_t>::max() / 1024ULL ? 0 : value_kb * 1024ULL;
}
std::uint64_t multiplyOrZero(std::uint64_t lhs, std::uint64_t rhs)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs)
        return 0;
    return lhs * rhs;
}
std::vector<std::size_t> parseCpuList(const std::string &cpu_list)
{
    std::vector<std::size_t> cpus;
    std::stringstream stream(cpu_list);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        const auto dash = token.find('-');
        if (dash == std::string::npos)
        {
            std::size_t cpu = 0;
            if (parseSizeT(token, &cpu))
            {
                cpus.push_back(cpu);
            }
            continue;
        }
        const std::string begin_str = trimCopy(token.substr(0, dash));
        const std::string end_str = trimCopy(token.substr(dash + 1));
        std::size_t begin = 0;
        std::size_t end = 0;
        if (!parseSizeT(begin_str, &begin) || !parseSizeT(end_str, &end) || begin > end)
        {
            continue;
        }
        for (std::size_t cpu = begin;; ++cpu)
        {
            cpus.push_back(cpu);
            if (cpu == end)
            {
                break;
            }
        }
    }
    std::ranges::sort(cpus);
    const auto [first, last] = std::ranges::unique(cpus);
    cpus.erase(first, last);
    return cpus;
}
std::uint64_t readMemtotalFromNode(const std::filesystem::path &path)
{
    std::ifstream stream(path);
    if (!stream.is_open())
    {
        return 0;
    }
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.rfind("MemTotal:", 0) != 0)
        {
            continue;
        }
        std::stringstream parser(line);
        std::string key;
        std::uint64_t value_kb = 0;
        parser >> key >> value_kb;
        return kilobytesToBytes(value_kb);
    }
    return 0;
}
ProbeValue<CpuTopology> probeCpuTopology()
{
    ProbeValue<CpuTopology> cpu;
    CpuTopology topology;
    const unsigned int logical = std::thread::hardware_concurrency();
    topology.logical_cores = logical == 0U ? 1U : static_cast<std::size_t>(logical);
#if defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::unordered_set<std::string> physical_cores;
    std::string model_name;
    if (cpuinfo.is_open())
    {
        std::string line;
        std::string physical_id = "0";
        std::string core_id;
        auto insert_physical_core = [&physical_cores, &physical_id, &core_id]() {
            if (core_id.empty())
            {
                return;
            }
            std::string physical_core_key;
            physical_core_key.reserve(physical_id.size() + 1 + core_id.size());
            physical_core_key.append(physical_id);
            physical_core_key.push_back(':');
            physical_core_key.append(core_id);
            physical_cores.insert(std::move(physical_core_key));
        };
        while (std::getline(cpuinfo, line))
        {
            if (line.empty())
            {
                insert_physical_core();
                physical_id = "0";
                core_id.clear();
                continue;
            }
            const auto sep = line.find(':');
            if (sep == std::string::npos)
            {
                continue;
            }
            const std::string key = trimCopy(line.substr(0, sep));
            const std::string value = trimCopy(line.substr(sep + 1));
            if (key == "model name" && model_name.empty())
            {
                model_name = value;
            }
            else if (key == "physical id")
            {
                physical_id = value;
            }
            else if (key == "core id" || (key == "processor" && core_id.empty()))
            {
                core_id = value;
            }
        }
        insert_physical_core();
    }
    topology.physical_cores =
        physical_cores.empty() ? topology.logical_cores : physical_cores.size();
    topology.model = model_name.empty() ? "unknown" : model_name;
#elif defined(__APPLE__)
    std::unordered_set<std::string> physical_cores;
    std::string model_name;
    {
        int mib[2] = {CTL_HW, HW_PER_CPU};
        int core_count = 0;
        std::size_t len = sizeof(core_count);
        if (sysctl(mib, 2, &core_count, &len, nullptr, 0) == 0 && core_count > 0)
        {
            topology.physical_cores = static_cast<std::size_t>(core_count);
        }
        else
        {
            topology.physical_cores = topology.logical_cores;
        }
    }
    {
        char buf[256]{};
        std::size_t len = sizeof(buf);
        if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0)
        {
            model_name = buf;
        }
    }
    topology.model = model_name.empty() ? "unknown" : model_name;
#else
    topology.physical_cores = topology.logical_cores;
    topology.model = "unknown";
#endif
    cpu.status = ProbeStatus::kOk;
    cpu.value = std::move(topology);
    return cpu;
}
ProbeValue<std::uint64_t> probeMemoryValue(bool available_memory)
{
    ProbeValue<std::uint64_t> out;
    std::uint64_t value = 0;
#if defined(__linux__)
    struct sysinfo info
    {};
    if (sysinfo(&info) != 0)
    {
        out.status = ProbeStatus::kError;
        out.diagnostics = "sysinfo failed";
        return out;
    }
    if (available_memory)
    {
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open())
        {
            std::string line;
            while (std::getline(meminfo, line))
            {
                if (line.rfind("MemAvailable:", 0) != 0)
                {
                    continue;
                }
                std::stringstream parser(line);
                std::string key;
                std::uint64_t value_kb = 0;
                parser >> key >> value_kb;
                value = kilobytesToBytes(value_kb);
                break;
            }
        }
        if (value == 0)
        {
            value = multiplyOrZero(static_cast<std::uint64_t>(info.freeram),
                                   static_cast<std::uint64_t>(info.mem_unit));
        }
    }
    else
    {
        value = multiplyOrZero(static_cast<std::uint64_t>(info.totalram),
                               static_cast<std::uint64_t>(info.mem_unit));
    }
#elif defined(__APPLE__)
    if (available_memory)
    {
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        vm_statistics64_data_t vm_stat;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                              reinterpret_cast<host_info64_t>(&vm_stat), &count) == KERN_SUCCESS)
        {
            value = static_cast<std::uint64_t>(vm_stat.free_count) *
                    static_cast<std::uint64_t>(vm_page_size);
        }
    }
    if (value == 0)
    {
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        std::uint64_t memsize = 0;
        std::size_t len = sizeof(memsize);
        if (sysctl(mib, 2, &memsize, &len, nullptr, 0) == 0)
        {
            value = memsize;
        }
    }
    if (value == 0)
    {
        out.status = ProbeStatus::kError;
        out.diagnostics = "memory probe failed on macOS";
        return out;
    }
#else
    out.status = ProbeStatus::kMissing;
    out.diagnostics = "memory probe not implemented on this platform";
    return out;
#endif
    out.status = ProbeStatus::kOk;
    out.value = value;
    return out;
}
ProbeValue<std::vector<NumaNodeInfo>> probeNumaNodes()
{
    ProbeValue<std::vector<NumaNodeInfo>> out;
    const std::filesystem::path nodeRoot("/sys/devices/system/node");
    std::error_code fs_error;
    if (!std::filesystem::exists(nodeRoot, fs_error) || fs_error)
    {
        out.status = ProbeStatus::kMissing;
        out.diagnostics = "numa node sysfs missing";
        return out;
    }
    std::vector<NumaNodeInfo> nodes;
    std::filesystem::directory_iterator entries(nodeRoot, fs_error);
    if (fs_error)
    {
        out.status = ProbeStatus::kError;
        out.diagnostics = "numa node sysfs iteration failed";
        return out;
    }
    const std::filesystem::directory_iterator end_entries;
    for (; entries != end_entries; entries.increment(fs_error))
    {
        if (fs_error)
        {
            break;
        }
        const auto &entry = *entries;
        std::error_code entry_error;
        if (!entry.is_directory(entry_error) || entry_error)
        {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("node", 0) != 0 || name.size() <= 4)
        {
            continue;
        }
        const std::string id_str = name.substr(4);
        std::size_t node_id = 0;
        if (!parseSizeT(id_str, &node_id))
        {
            continue;
        }
        NumaNodeInfo node;
        node.node_id = node_id;
        const auto cpulist_path = entry.path() / "cpulist";
        std::ifstream cpusStream(cpulist_path);
        if (cpusStream.is_open())
        {
            std::string cpu_line;
            std::getline(cpusStream, cpu_line);
            node.cpu_ids = parseCpuList(cpu_line);
        }
        node.total_memory_bytes = readMemtotalFromNode(entry.path() / "meminfo");
        nodes.push_back(std::move(node));
    }
    if (fs_error && nodes.empty())
    {
        out.status = ProbeStatus::kError;
        out.diagnostics = "numa node sysfs iteration failed";
        return out;
    }
    if (nodes.empty())
    {
        out.status = ProbeStatus::kMissing;
        out.diagnostics = "no NUMA node entries detected";
        return out;
    }
    std::ranges::sort(nodes, {}, &NumaNodeInfo::node_id);
    out.status = ProbeStatus::kOk;
    out.value = std::move(nodes);
    return out;
}
ProbeValue<std::vector<GpuDeviceInfo>> probeGpuDevices()
{
    ProbeValue<std::vector<GpuDeviceInfo>> out;
#if defined(NERVE_HAS_CUDA_RUNTIME) || defined(NERVE_HAS_CUDA)
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess)
    {
        out.status = ProbeStatus::kMissing;
        out.diagnostics = cudaGetErrorString(count_status);
        return out;
    }
    if (device_count <= 0)
    {
        out.status = ProbeStatus::kMissing;
        out.diagnostics = "cuda runtime available but no devices";
        return out;
    }
    int prior_device = 0;
    const cudaError_t current_device_status = cudaGetDevice(&prior_device);
    if (current_device_status != cudaSuccess)
    {
        prior_device = 0;
    }
    std::vector<GpuDeviceInfo> devices;
    devices.reserve(static_cast<std::size_t>(device_count));
    for (int index = 0; index < device_count; ++index)
    {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, index) != cudaSuccess)
        {
            continue;
        }
        GpuDeviceInfo device;
        device.device_id = index;
        device.name = props.name;
        device.compute_capability_major = props.major;
        device.compute_capability_minor = props.minor;
        device.total_memory_bytes = static_cast<std::uint64_t>(props.totalGlobalMem);
        device.supports_tensor_cores = props.major >= 7;
        if (cudaSetDevice(index) == cudaSuccess)
        {
            std::size_t freeMemory = 0;
            std::size_t total_memory = 0;
            if (cudaMemGetInfo(&freeMemory, &total_memory) == cudaSuccess)
            {
                device.free_memory_bytes = freeMemory;
                device.total_memory_bytes = total_memory;
            }
        }
        devices.push_back(std::move(device));
    }
    if (current_device_status == cudaSuccess)
    {
        cudaSetDevice(prior_device);
    }
    if (devices.empty())
    {
        out.status = ProbeStatus::kError;
        out.diagnostics = "failed to read CUDA device properties";
        return out;
    }
    out.status = ProbeStatus::kOk;
    out.value = std::move(devices);
    return out;
#else
    out.status = ProbeStatus::kMissing;
    out.diagnostics = "cuda runtime not compiled";
    return out;
#endif
}
} // namespace
std::string probe_status_to_string(ProbeStatus status)
{
    switch (status)
    {
        case ProbeStatus::kOk:
            return "ok";
        case ProbeStatus::kMissing:
            return "missing";
        case ProbeStatus::kError:
            return "error";
        default:
            return "unknown";
    }
}
HardwareSnapshot collectHardwareSnapshot()
{
    HardwareSnapshot snapshot;
    snapshot.collected_unix_ms = unixTimeMs();
    snapshot.cpu = probeCpuTopology();
    snapshot.total_memory_bytes = probeMemoryValue(false);
    snapshot.available_memory_bytes = probeMemoryValue(true);
    snapshot.numa_nodes = probeNumaNodes();
    snapshot.gpus = probeGpuDevices();
    if (!snapshot.cpu.ok())
    {
        snapshot.diagnostics.push_back("cpu:" + snapshot.cpu.diagnostics);
    }
    if (!snapshot.total_memory_bytes.ok())
    {
        snapshot.diagnostics.push_back("total_memory:" + snapshot.total_memory_bytes.diagnostics);
    }
    if (!snapshot.available_memory_bytes.ok())
    {
        snapshot.diagnostics.push_back("available_memory:" +
                                       snapshot.available_memory_bytes.diagnostics);
    }
    if (snapshot.numa_nodes.status == ProbeStatus::kError)
    {
        snapshot.diagnostics.push_back("numa:" + snapshot.numa_nodes.diagnostics);
    }
    if (snapshot.gpus.status == ProbeStatus::kError)
    {
        snapshot.diagnostics.push_back("gpu:" + snapshot.gpus.diagnostics);
    }
    return snapshot;
}
std::string getHardwareFingerprint(const HardwareSnapshot &snapshot)
{
    std::ostringstream canonical;
    canonical << probe_status_to_string(snapshot.cpu.status) << '|';
    canonical << snapshot.cpu.value.logical_cores << '|';
    canonical << snapshot.cpu.value.physical_cores << '|';
    canonical << snapshot.cpu.value.model << '|';
    canonical << probe_status_to_string(snapshot.total_memory_bytes.status) << '|';
    canonical << snapshot.total_memory_bytes.value << '|';
    canonical << probe_status_to_string(snapshot.numa_nodes.status) << '|';
    canonical << snapshot.numa_nodes.value.size() << '|';
    canonical << probe_status_to_string(snapshot.gpus.status) << '|';
    canonical << snapshot.gpus.value.size();
    for (const auto &gpu : snapshot.gpus.value)
    {
        canonical << '|' << gpu.device_id << ':' << gpu.name << ':' << gpu.compute_capability_major
                  << '.' << gpu.compute_capability_minor << ':' << gpu.total_memory_bytes;
    }
    const std::uint64_t hash = fnv1aHash(canonical.str());
    std::ostringstream fingerprint;
    fingerprint << std::hex << std::setfill('0') << std::setw(16) << hash;
    return fingerprint.str();
}
bool has_cuda_gpu(const HardwareSnapshot &snapshot)
{
    return snapshot.gpus.status == ProbeStatus::kOk && !snapshot.gpus.value.empty();
}
} // namespace nerve::runtime
