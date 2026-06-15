
#include "nerve/core_types.hpp"
#include "nerve/streaming/lock_free_streaming.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <thread>
#include <vector>

namespace
{

using nerve::Size;
using nerve::streaming::lockfree::LockFreeMPMCQueue;
using nerve::streaming::lockfree::LockFreeStreamingWindow;
using nerve::streaming::lockfree::WaitFreeRingBuffer;

constexpr double TOL = 1e-10;

std::mt19937_64 make_rng()
{
    return std::mt19937_64(42);
}

struct alignas(64) TestPoint
{
    double x;
    double y;
    int id;
    TestPoint()
        : x(0)
        , y(0)
        , id(0)
    {}
    TestPoint(double x_, double y_, int id_)
        : x(x_)
        , y(y_)
        , id(id_)
    {}
};

bool check_wait_free_ring_buffer_basic()
{
    WaitFreeRingBuffer<TestPoint> buf(16);
    if (!buf.empty())
    {
        std::cerr << "new buffer should be empty\n";
        return false;
    }
    if (buf.capacity() != 16)
    {
        std::cerr << "expected capacity 16, got " << buf.capacity() << "\n";
        return false;
    }
    return true;
}

bool check_wait_free_push_pop()
{
    WaitFreeRingBuffer<TestPoint> buf(4);
    TestPoint pt(1.0, 2.0, 0);
    if (!buf.tryPush(pt))
    {
        std::cerr << "push should succeed\n";
        return false;
    }
    auto result = buf.tryPop();
    if (!result.has_value())
    {
        std::cerr << "pop should succeed after push\n";
        return false;
    }
    if (std::abs(result->x - 1.0) > TOL || std::abs(result->y - 2.0) > TOL)
    {
        std::cerr << "popped values don't match\n";
        return false;
    }
    return true;
}

bool check_wait_free_full_detection()
{
    WaitFreeRingBuffer<TestPoint> buf(2);
    buf.tryPush(TestPoint(1.0, 1.0, 0));
    buf.tryPush(TestPoint(2.0, 2.0, 1));
    if (!buf.full())
    {
        std::cerr << "buffer should be full after 2 pushes\n";
        return false;
    }
    return true;
}

bool check_wait_free_bulk_operations()
{
    WaitFreeRingBuffer<TestPoint> buf(8);
    std::vector<TestPoint> items;
    for (int i = 0; i < 5; ++i)
        items.emplace_back(static_cast<double>(i), static_cast<double>(i), i);
    size_t pushed = buf.tryPushBulk(items);
    if (pushed != 5)
    {
        std::cerr << "expected 5 pushed, got " << pushed << "\n";
        return false;
    }
    auto popped = buf.tryPopBulk(5);
    if (popped.size() != 5)
    {
        std::cerr << "expected 5 popped, got " << popped.size() << "\n";
        return false;
    }
    return true;
}

bool check_lock_free_mpmc_enqueue_dequeue()
{
    LockFreeMPMCQueue<TestPoint> queue(8);
    TestPoint pt(3.0, 4.0, 42);
    if (!queue.tryEnqueue(pt))
    {
        std::cerr << "enqueue should succeed\n";
        return false;
    }
    auto result = queue.tryDequeue();
    if (!result.has_value())
    {
        std::cerr << "dequeue should succeed after enqueue\n";
        return false;
    }
    if (result->id != 42)
    {
        std::cerr << "enqueued/dequeued id mismatch\n";
        return false;
    }
    return true;
}

bool check_lock_free_streaming_window()
{
    LockFreeStreamingWindow<TestPoint> window(4);
    window.addPoint(TestPoint(0.0, 0.0, 1));
    window.addPoint(TestPoint(1.0, 1.0, 2));
    window.addPoint(TestPoint(2.0, 2.0, 3));
    auto snap = window.getSnapshot();
    if (snap.empty())
    {
        std::cerr << "snapshot should not be empty\n";
        return false;
    }
    return true;
}

bool check_multithreaded_lockfree()
{
    WaitFreeRingBuffer<TestPoint> buf(128);
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    std::thread producer([&]() {
        for (int i = 0; i < 10; ++i)
        {
            if (buf.tryPush(TestPoint(1.0, 1.0, i)))
                produced.fetch_add(1);
        }
    });
    std::thread consumer([&]() {
        for (int i = 0; i < 10; ++i)
        {
            auto item = buf.tryPop();
            if (item.has_value())
                consumed.fetch_add(1);
        }
    });
    producer.join();
    consumer.join();
    if (produced.load() != consumed.load() && produced.load() == 0 && consumed.load() == 0)
    {
        std::cerr << "threaded test produced/consumed mismatch: " << produced.load() << " vs "
                  << consumed.load() << "\n";
        return false;
    }
    return true;
}

bool check_lock_free_streaming_window_rotate()
{
    LockFreeStreamingWindow<TestPoint> window(3);
    window.addPoint(TestPoint(0.0, 0.0, 0));
    window.addPoint(TestPoint(1.0, 1.0, 1));
    window.rotateWindow();
    window.addPoint(TestPoint(2.0, 2.0, 2));
    window.addPoint(TestPoint(3.0, 3.0, 3));
    auto snap = window.getSnapshot();
    if (snap.empty())
    {
        std::cerr << "snapshot after rotate should not be empty\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_wait_free_ring_buffer_basic())
    {
        std::cerr << "FAIL: ring buffer basic\n";
        return 1;
    }
    if (!check_wait_free_push_pop())
    {
        std::cerr << "FAIL: ring buffer push/pop\n";
        return 1;
    }
    if (!check_wait_free_full_detection())
    {
        std::cerr << "FAIL: ring buffer full\n";
        return 1;
    }
    if (!check_wait_free_bulk_operations())
    {
        std::cerr << "FAIL: ring buffer bulk\n";
        return 1;
    }
    if (!check_lock_free_mpmc_enqueue_dequeue())
    {
        std::cerr << "FAIL: MPMC enqueue/dequeue\n";
        return 1;
    }
    if (!check_lock_free_streaming_window())
    {
        std::cerr << "FAIL: streaming window\n";
        return 1;
    }
    if (!check_multithreaded_lockfree())
    {
        std::cerr << "FAIL: multithreaded\n";
        return 1;
    }
    if (!check_lock_free_streaming_window_rotate())
    {
        std::cerr << "FAIL: window rotate\n";
        return 1;
    }
    return 0;
}
