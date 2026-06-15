#pragma once
#ifdef NERVE_USE_NVSHMEM

#include <cuda_runtime.h>
#include <nvshmem.h>

#include <cstddef>

namespace nerve::distributed
{

class NvshmemBridge
{
public:
    void init();
    void finalize();

    template <typename T>
    void put(T *dest, const T *source, size_t count, int pe)
    {
        if (!initialized_)
            return;
        nvshmem_putmem(dest, source, count * sizeof(T), pe);
    }

    template <typename T>
    void get(T *dest, const T *source, size_t count, int pe)
    {
        if (!initialized_)
            return;
        nvshmem_getmem(dest, source, count * sizeof(T), pe);
    }

    template <typename T>
    void reduce(T *dest, const T *source, size_t count, int pe)
    {
        if (!initialized_)
            return;
        nvshmem_putmem(dest, source, count * sizeof(T), pe);
    }

    void barrier();

    template <typename T>
    T *symmetric_malloc(size_t count)
    {
        if (!initialized_)
            return nullptr;
        return static_cast<T *>(nvshmem_malloc(count * sizeof(T)));
    }

    void free(void *ptr);

private:
    bool initialized_ = false;
    int my_pe_ = -1;
    int n_pes_ = 0;
};

} // namespace nerve::distributed

#else // !NERVE_USE_NVSHMEM

#include <cstddef>
#include <cstdlib>

namespace nerve::distributed
{

class NvshmemBridge
{
public:
    void init();
    void finalize();

    template <typename T>
    void put(T *, const T *, size_t, int)
    {}

    template <typename T>
    void get(T *, const T *, size_t, int)
    {}

    template <typename T>
    void reduce(T *, const T *, size_t, int)
    {}

    void barrier();

    template <typename T>
    T *symmetric_malloc(size_t count)
    {
        return static_cast<T *>(std::malloc(count * sizeof(T)));
    }

    void free(void *ptr);

private:
    bool initialized_ = false;
};

} // namespace nerve::distributed

#endif // NERVE_USE_NVSHMEM
