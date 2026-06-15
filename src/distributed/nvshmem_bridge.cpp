#include "nerve/distributed/nvshmem_bridge.hpp"

#ifdef NERVE_USE_NVSHMEM

#include <stdexcept>

namespace nerve::distributed
{

void NvshmemBridge::init()
{
    if (initialized_)
        return;
    nvshmem_init();
    my_pe_ = nvshmem_my_pe();
    n_pes_ = nvshmem_n_pes();
    initialized_ = true;
}

void NvshmemBridge::finalize()
{
    if (!initialized_)
        return;
    nvshmem_finalize();
    initialized_ = false;
    my_pe_ = -1;
    n_pes_ = 0;
}

void NvshmemBridge::barrier()
{
    if (!initialized_)
        return;
    nvshmem_barrier_all();
}

void NvshmemBridge::free(void *ptr)
{
    if (!initialized_ || ptr == nullptr)
        return;
    nvshmem_free(ptr);
}

} // namespace nerve::distributed

#else // !NERVE_USE_NVSHMEM

#include <cstdlib>

namespace nerve::distributed
{

void NvshmemBridge::init()
{
    initialized_ = true;
}

void NvshmemBridge::finalize()
{
    initialized_ = false;
}

void NvshmemBridge::barrier() {}

void NvshmemBridge::free(void *ptr)
{
    std::free(ptr);
}

} // namespace nerve::distributed

#endif // NERVE_USE_NVSHMEM
