#include "nerve/core.hpp"

#include <cuda_runtime.h>
#if __has_include(<nccl.h>)
#include <nccl.h>
#define NERVE_HAS_NCCL 1
#else
#define NERVE_HAS_NCCL 0
using ncclComm_t = void *;
using ncclRedOp_t = int;
using ncclDataType_t = int;
constexpr ncclRedOp_t ncclSum = 0;
constexpr ncclDataType_t ncclFloat = 0;
constexpr ncclDataType_t ncclDouble = 1;
constexpr ncclDataType_t ncclInt = 2;
#endif
#include <algorithm>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace nerve
{
namespace gpu
{
namespace multi
{

struct GPUDevice
{
    int id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability;
    bool has_nvlink;
    std::vector<int> peer_access; // GPUs accessible via P2P
};

class MultiGPUManager
{
public:
    MultiGPUManager()
    {
        int device_count = 0;
        cudaError_t count_err = cudaGetDeviceCount(&device_count);
        if (count_err != cudaSuccess || device_count <= 0)
        {
            return;
        }

        for (int i = 0; i < device_count; ++i)
        {
            GPUDevice dev;
            dev.id = i;

            cudaSetDevice(i);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            dev.name = prop.name;
            dev.compute_capability = prop.major * 10 + prop.minor;

            size_t free, total;
            cudaMemGetInfo(&free, &total);
            dev.total_memory = total;
            dev.free_memory = free;

            for (int j = 0; j < device_count; ++j)
            {
                if (i != j)
                {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, i, j);
                    if (can_access)
                    {
                        dev.peer_access.push_back(j);
                        cudaDeviceEnablePeerAccess(j, 0);
                    }
                }
            }

            dev.has_nvlink = !dev.peer_access.empty();

            devices_.push_back(dev);
        }
#if NERVE_HAS_NCCL
        if (ncclCommInitAll(&nccl_comm_, device_count, nullptr) == ncclSuccess)
        {
            nccl_initialized_ = true;
        }
#endif
    }

    ~MultiGPUManager()
    {
#if NERVE_HAS_NCCL
        if (nccl_initialized_)
        {
            ncclCommDestroy(nccl_comm_);
        }
#endif
    }

    int getDeviceCount() const { return static_cast<int>(devices_.size()); }

    const GPUDevice &getDevice(int id) const
    {
        if (devices_.empty())
        {
            throw std::out_of_range("No GPU devices are available");
        }
        const int idx = std::clamp(id, 0, static_cast<int>(devices_.size()) - 1);
        return devices_[idx];
    }
    int selectGPU()
    {
        if (devices_.empty())
        {
            return -1;
        }
        int best_gpu = 0;
        size_t max_free = 0;

        for (int i = 0; i < static_cast<int>(devices_.size()); ++i)
        {
            cudaSetDevice(i);
            size_t free, total;
            cudaMemGetInfo(&free, &total);

            if (free > max_free)
            {
                max_free = free;
                best_gpu = i;
            }
        }

        return best_gpu;
    }
    std::vector<size_t> distributeWork(size_t total_work)
    {
        std::vector<size_t> distribution;
        if (devices_.empty())
        {
            return distribution;
        }
        size_t total_free = 0;
        for (const auto &dev : devices_)
        {
            total_free += dev.free_memory;
        }
        if (total_free == 0)
        {
            distribution.assign(devices_.size(), total_work / devices_.size());
            distribution.front() += total_work % devices_.size();
            return distribution;
        }

        for (const auto &dev : devices_)
        {
            size_t share = (dev.free_memory * total_work) / total_free;
            distribution.push_back(share);
        }

        size_t distributed = 0;
        for (auto &d : distribution)
        {
            distributed += d;
        }
        distribution[0] += total_work - distributed;

        return distribution;
    }
    template <typename T>
    void allReduce(T *data, size_t count, ncclRedOp_t op = ncclSum)
    {
#if NERVE_HAS_NCCL
        if (!nccl_initialized_ || data == nullptr || count == 0)
        {
            return;
        }
        ncclDataType_t type;
        if (std::is_same_v<T, float>)
            type = ncclFloat;
        else if (std::is_same_v<T, double>)
            type = ncclDouble;
        else if (std::is_same_v<T, int>)
            type = ncclInt;
        else
            return;

        ncclAllReduce(data, data, count, type, op, nccl_comm_, nullptr);
#else
        (void)data;
        (void)count;
        (void)op;
#endif
    }
    template <typename T>
    void broadcast(T *data, size_t count, int root = 0)
    {
#if NERVE_HAS_NCCL
        if (!nccl_initialized_ || data == nullptr || count == 0)
        {
            return;
        }
        ncclDataType_t type;
        if (std::is_same_v<T, float>)
            type = ncclFloat;
        else if (std::is_same_v<T, double>)
            type = ncclDouble;
        else if (std::is_same_v<T, int>)
            type = ncclInt;
        else
            return;

        ncclBroadcast(data, data, count, type, root, nccl_comm_, nullptr);
#else
        (void)data;
        (void)count;
        (void)root;
#endif
    }
    void p2pCopy(int src_gpu, void *src_ptr, int dst_gpu, void *dst_ptr, size_t size)
    {
        cudaSetDevice(src_gpu);
        cudaMemcpyPeer(dst_ptr, dst_gpu, src_ptr, src_gpu, size);
    }
    bool canAccessPeer(int src, int dst)
    {
        if (src == dst)
            return true;

        int can_access;
        cudaDeviceCanAccessPeer(&can_access, src, dst);
        return can_access == 1;
    }

private:
    std::vector<GPUDevice> devices_;
    ncclComm_t nccl_comm_;
    bool nccl_initialized_ = false;
};

class GPUDirectStorage
{
public:
    GPUDirectStorage() {}

    bool readToGPU(const std::string &filename, void *gpu_ptr, size_t offset, size_t size)
    {
#ifdef HAVE_CUFILE
        return readToGPU_cuFile(filename, gpu_ptr, offset, size);
#else
        return readToGPU_hostStaging(filename, gpu_ptr, offset, size);
#endif
    }

private:
    bool readToGPU_hostStaging(const std::string &filename, void *gpu_ptr, size_t offset,
                               size_t size)
    {
        char *pinned_buffer;
        if (cudaMallocHost(&pinned_buffer, size) != cudaSuccess)
        {
            return false;
        }

        FILE *f = fopen(filename.c_str(), "rb");
        if (!f)
        {
            cudaFreeHost(pinned_buffer);
            return false;
        }

        fseek(f, offset, SEEK_SET);
        size_t read_bytes = fread(pinned_buffer, 1, size, f);
        fclose(f);

        if (read_bytes != size)
        {
            cudaFreeHost(pinned_buffer);
            return false;
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(gpu_ptr, pinned_buffer, size, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        cudaFreeHost(pinned_buffer);
        return true;
    }

#ifdef HAVE_CUFILE
    bool readToGPU_cuFile(const std::string &filename, void *gpu_ptr, size_t offset, size_t size)
    {
        return readToGPU_hostStaging(filename, gpu_ptr, offset, size);
    }
#endif

public:
    bool batchReadToGPU(const std::vector<std::string> &filenames,
                        const std::vector<void *> &gpu_ptrs, const std::vector<size_t> &sizes)
    {
        if (filenames.size() != gpu_ptrs.size() || filenames.size() != sizes.size())
        {
            throw std::invalid_argument("batchReadToGPU input vectors must have matching sizes");
        }
        for (size_t i = 0; i < filenames.size(); ++i)
        {
            if (!readToGPU(filenames[i], gpu_ptrs[i], 0, sizes[i]))
            {
                return false;
            }
        }
        return true;
    }
};

class LoadBalancer
{
public:
    struct Task
    {
        int id;
        size_t workload;
        int assigned_gpu;
    };

    explicit LoadBalancer(int num_gpus)
        : num_gpus_(num_gpus)
    {
        if (num_gpus_ <= 0)
        {
            throw std::invalid_argument("num_gpus must be positive");
        }
    }

    std::vector<Task> assignTasks(const std::vector<size_t> &workloads)
    {
        std::vector<Task> tasks;
        std::vector<size_t> gpu_load(num_gpus_, 0);

        std::vector<std::pair<size_t, int>> sorted;
        for (size_t i = 0; i < workloads.size(); ++i)
        {
            sorted.push_back({workloads[i], static_cast<int>(i)});
        }
        std::ranges::sort(sorted, std::greater{});

        for (const auto &[work, id] : sorted)
        {
            int min_gpu = 0;
            size_t min_load = gpu_load[0];

            for (int i = 1; i < num_gpus_; ++i)
            {
                if (gpu_load[i] < min_load)
                {
                    min_load = gpu_load[i];
                    min_gpu = i;
                }
            }

            tasks.push_back({id, work, min_gpu});
            gpu_load[min_gpu] += work;
        }

        return tasks;
    }

    std::vector<Task> rebalance(const std::vector<Task> &current,
                                const std::vector<double> &execution_times)
    {
        if (current.empty() || execution_times.empty())
        {
            return current;
        }
        std::vector<double> gpu_avg_time(num_gpus_, 0.0);
        std::vector<int> gpu_task_count(num_gpus_, 0);

        for (size_t i = 0; i < current.size(); ++i)
        {
            int gpu = current[i].assigned_gpu;
            if (i < execution_times.size() && gpu >= 0 && gpu < num_gpus_)
            {
                gpu_avg_time[gpu] += execution_times[i];
                gpu_task_count[gpu]++;
            }
        }

        for (int i = 0; i < num_gpus_; ++i)
        {
            if (gpu_task_count[i] > 0)
            {
                gpu_avg_time[i] /= gpu_task_count[i];
            }
        }

        int slowest_gpu = 0;
        int fastest_gpu = 0;
        double max_time = gpu_avg_time[0];
        double min_time = gpu_avg_time[0];

        for (int i = 1; i < num_gpus_; ++i)
        {
            if (gpu_avg_time[i] > max_time)
            {
                max_time = gpu_avg_time[i];
                slowest_gpu = i;
            }
            if (gpu_avg_time[i] < min_time && gpu_task_count[i] > 0)
            {
                min_time = gpu_avg_time[i];
                fastest_gpu = i;
            }
        }
        if (max_time > min_time * 1.2 && gpu_task_count[slowest_gpu] > 1)
        {
            std::vector<Task> rebalanced = current;

            for (auto &task : rebalanced)
            {
                if (task.assigned_gpu == slowest_gpu)
                {
                    task.assigned_gpu = fastest_gpu;
                    break;
                }
            }

            return rebalanced;
        }

        return current;
    }

private:
    int num_gpus_;
};

} // namespace multi
} // namespace gpu
} // namespace nerve
