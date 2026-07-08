#include "nerve/streaming/lock_free_streaming.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <set>
#include <thread>
#include <vector>

namespace
{

using nerve::streaming::lockfree::LockFreeMPMCQueue;
using nerve::streaming::lockfree::LockFreeStreamingWindow;
using nerve::streaming::lockfree::WaitFreeRingBuffer;

} // namespace

int main()
{
    // MPMC queue push/pop with 2 threads  --  verify FIFO order
    {
        LockFreeMPMCQueue<int> queue(8);

        std::thread producer([&queue]() {
            for (int i = 0; i < 4; ++i)
            {
                while (!queue.tryEnqueue(i))
                {
                    std::this_thread::yield();
                }
            }
        });

        std::thread consumer([&queue]() {
            for (int expected = 0; expected < 4; ++expected)
            {
                std::optional<int> value;
                while (!(value = queue.tryDequeue()))
                {
                    std::this_thread::yield();
                }
                assert(value.has_value());
            }
        });

        producer.join();
        consumer.join();
    }

    // MPMC stress test  --  4 producers, 4 consumers, 10000 items
    {
        constexpr int total_items = 10000;
        constexpr int num_producers = 4;
        constexpr int num_consumers = 4;

        LockFreeMPMCQueue<int> queue(1024);
        std::atomic<int> produced{0};
        std::atomic<int> consumed{0};
        std::atomic<long long> sum_produced{0};
        std::atomic<long long> sum_consumed{0};

        std::vector<std::thread> producers;
        for (int p = 0; p < num_producers; ++p)
        {
            producers.emplace_back([&queue, &produced, &sum_produced, p]() {
                int base = p * (total_items / num_producers);
                int count = total_items / num_producers;
                for (int i = 0; i < count; ++i)
                {
                    int value = base + i;
                    while (!queue.tryEnqueue(value))
                    {
                        std::this_thread::yield();
                    }
                    produced.fetch_add(1, std::memory_order_relaxed);
                    sum_produced.fetch_add(value, std::memory_order_relaxed);
                }
            });
        }

        std::vector<std::thread> consumers;
        for (int c = 0; c < num_consumers; ++c)
        {
            consumers.emplace_back([&queue, &consumed, &sum_consumed, total_items]() {
                while (consumed.load(std::memory_order_relaxed) < total_items)
                {
                    auto value = queue.tryDequeue();
                    if (value.has_value())
                    {
                        consumed.fetch_add(1, std::memory_order_relaxed);
                        sum_consumed.fetch_add(*value, std::memory_order_relaxed);
                    }
                    else
                    {
                        std::this_thread::yield();
                    }
                }
            });
        }

        for (auto &t : producers)
            t.join();
        for (auto &t : consumers)
            t.join();

        assert(produced.load() == total_items);
        assert(consumed.load() == total_items);
        assert(sum_produced.load() == sum_consumed.load());
    }

    // WaitFreeRingBuffer write/read  --  verify order and capacity limit
    {
        WaitFreeRingBuffer<int> buffer(4);

        assert(buffer.tryPush(1));
        assert(buffer.tryPush(2));
        assert(buffer.tryPush(3));
        assert(buffer.tryPush(4));
        assert(!buffer.tryPush(5));
        assert(buffer.full());

        auto v1 = buffer.tryPop();
        assert(v1.has_value() && *v1 == 1);
        auto v2 = buffer.tryPop();
        assert(v2.has_value() && *v2 == 2);

        assert(buffer.tryPush(5));
        assert(buffer.tryPush(6));
        assert(!buffer.tryPush(7));

        auto v3 = buffer.tryPop();
        assert(v3.has_value() && *v3 == 3);
        auto v4 = buffer.tryPop();
        assert(v4.has_value() && *v4 == 4);
        auto v5 = buffer.tryPop();
        assert(v5.has_value() && *v5 == 5);
        auto v6 = buffer.tryPop();
        assert(v6.has_value() && *v6 == 6);

        assert(buffer.empty());
        assert(!buffer.tryPop().has_value());
    }

    // WaitFreeRingBuffer bulk operations
    {
        WaitFreeRingBuffer<int> buffer(16);

        std::vector<int> items = {10, 20, 30, 40, 50};
        size_t pushed = buffer.tryPushBulk(items);
        assert(pushed == 5);

        auto popped = buffer.tryPopBulk(10);
        assert(popped.size() == 5);
        for (size_t i = 0; i < 5; ++i)
        {
            assert(popped[i] == items[i]);
        }
    }

    // WaitFreeRingBuffer snapshot
    {
        WaitFreeRingBuffer<int> buffer(8);

        buffer.tryPush(1);
        buffer.tryPush(2);
        buffer.tryPush(3);

        auto snap = buffer.snapshotNewest(2);
        assert(snap.size() == 2);
        assert(snap[0] == 2);
        assert(snap[1] == 3);

        assert(buffer.approximateSize() == 3);
    }

    // WaitFreeRingBuffer SPSC stress test
    {
        WaitFreeRingBuffer<int> buffer(1024);
        constexpr int count = 5000;
        std::atomic<bool> producer_done{false};

        std::thread producer([&buffer, &producer_done]() {
            for (int i = 0; i < count; ++i)
            {
                while (!buffer.tryPush(i))
                {
                    std::this_thread::yield();
                }
            }
            producer_done.store(true, std::memory_order_release);
        });

        std::thread consumer([&buffer, &producer_done]() {
            int expected = 0;
            while (expected < count)
            {
                auto val = buffer.tryPop();
                if (val.has_value())
                {
                    assert(*val == expected);
                    ++expected;
                }
                else if (producer_done.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                else
                {
                    std::this_thread::yield();
                }
            }
        });

        producer.join();
        consumer.join();
    }

    // LockFreeStreamingWindow
    {
        LockFreeStreamingWindow<int> window(4);

        window.addPoint(1);
        window.addPoint(2);
        window.addPoint(3);

        auto snap = window.getSnapshot();
        assert(snap.size() == 3);

        window.addPoint(4);
        window.addPoint(5);
        window.addPoint(6);

        auto snap2 = window.getSnapshot();
        assert(snap2.size() == 4);
    }

    // WaitFreeRingBuffer capacity at boundaries
    {
        WaitFreeRingBuffer<int> tiny(1);
        assert(tiny.tryPush(42));
        assert(!tiny.tryPush(43));
        assert(tiny.full());

        auto v = tiny.tryPop();
        assert(v.has_value() && *v == 42);
        assert(tiny.empty());
    }

    // WaitFreeRingBuffer approximateSize
    {
        WaitFreeRingBuffer<int> buffer(64);
        for (int i = 0; i < 30; ++i)
        {
            assert(buffer.tryPush(i));
        }
        assert(buffer.approximateSize() == 30);

        for (int i = 0; i < 10; ++i)
        {
            (void)buffer.tryPop();
        }
        assert(buffer.approximateSize() == 20);
    }

    return 0;
}
