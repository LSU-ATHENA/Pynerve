#include "distributed_wire_format.hpp"
#include "nerve/distributed/mpi_persistence.hpp"

#include <chrono>
#include <cstdint>
#include <thread>

namespace nerve::distributed
{

WorkStealingScheduler::WorkStealingScheduler(int, int)
    : shutdown_(false)
{}

void WorkStealingScheduler::submit_work(std::function<void()> work)
{
    std::lock_guard<std::mutex> lock(work_queue_mutex_);
    local_work_queue_.push(std::move(work));
}

void WorkStealingScheduler::run()
{
    while (!shutdown_)
    {
        std::function<void()> work;
        {
            std::lock_guard<std::mutex> lock(work_queue_mutex_);
            if (!local_work_queue_.empty())
            {
                work = std::move(local_work_queue_.front());
                local_work_queue_.pop();
            }
        }
        if (work)
        {
            work();
            continue;
        }
        if (should_terminate())
        {
            break;
        }
        if (steal_work())
        {
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void WorkStealingScheduler::shutdown()
{
    shutdown_ = true;
}

bool WorkStealingScheduler::steal_work()
{
    // Work-stealing transport is not enabled in this scheduler:
    // tasks are opaque std::function instances and are not serializable.
    // Returning false here keeps execution local and avoids deadlock-prone
    // cross-rank handshakes with non-portable function payloads.
    return false;
}

bool WorkStealingScheduler::should_terminate()
{
    std::lock_guard<std::mutex> lock(work_queue_mutex_);
    return local_work_queue_.empty();
}

void WorkStealingScheduler::execute_stolen_work(const std::vector<uint8_t> &data)
{
    // Accept a minimal protocol for explicit tests/tools:
    // payload = repeated little-endian int32 values, each meaning "sleep N us".
    if (data.size() < detail::kUint32WireSize)
    {
        return;
    }
    std::size_t offset = 0;
    while (offset + detail::kUint32WireSize <= data.size())
    {
        std::int32_t micros = 0;
        if (!detail::readInt32LittleEndian(data, &offset, &micros))
        {
            return;
        }
        if (micros <= 0)
        {
            continue;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(micros));
    }
}

} // namespace nerve::distributed
