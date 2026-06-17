#include "nerve/distributed/nvshmem_bridge.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#ifdef NERVE_HAS_MPI

namespace
{

using nerve::distributed::NvshmemBridge;

#ifdef NERVE_HAS_NVSHMEM

bool check_bridge_init()
{
    NvshmemBridge bridge;
    bridge.init();
    bridge.finalize();
    return true;
}

bool check_memory_alloc_dealloc()
{
    NvshmemBridge bridge;
    bridge.init();
    int *ptr = bridge.symmetric_malloc<int>(64);
    if (ptr == nullptr)
    {
        std::cerr << "symmetric_malloc returned null\n";
        bridge.finalize();
        return false;
    }
    bridge.free(ptr);
    bridge.finalize();
    return true;
}

bool check_peer_access_config()
{
    NvshmemBridge bridge;
    bridge.init();
    bridge.barrier();
    bridge.finalize();
    return true;
}

#else

bool check_nvshmem_noop()
{
    NvshmemBridge bridge;
    bridge.init();
    bridge.finalize();
    return true;
}

bool check_no_nvshmem_malloc()
{
    NvshmemBridge bridge;
    bridge.init();
    int *ptr = bridge.symmetric_malloc<int>(32);
    if (ptr == nullptr)
    {
        std::cerr << "noop malloc returned null\n";
        bridge.finalize();
        return false;
    }
    bridge.free(ptr);
    bridge.finalize();
    return true;
}

#endif

} // namespace

int main()
{
#ifdef NERVE_HAS_NVSHMEM
    if (!check_bridge_init())
    {
        std::cerr << "FAIL: bridge_init\n";
        return 1;
    }
    if (!check_memory_alloc_dealloc())
    {
        std::cerr << "FAIL: memory_alloc_dealloc\n";
        return 1;
    }
    if (!check_peer_access_config())
    {
        std::cerr << "FAIL: peer_access_config\n";
        return 1;
    }
#else
    if (!check_nvshmem_noop())
    {
        std::cerr << "FAIL: no_nvshmem_noop\n";
        return 1;
    }
    if (!check_no_nvshmem_malloc())
    {
        std::cerr << "FAIL: no_nvshmem_malloc\n";
        return 1;
    }
#endif
    return 0;
}

#else
int main()
{
    return 0;
}
#endif