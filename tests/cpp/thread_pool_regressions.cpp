#include "threading/concurrent_vector.hpp"
#include "threading/thread_pool.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <thread>

int main()
{
    try
    {
        nerve::threading::ConcurrentVector<int> values;
        assert(values.maxSize() > 0);
        assert(values.begin() == values.end());
        bool rejected_empty_front = false;
        try
        {
            (void)values.front();
        }
        catch (const std::out_of_range &)
        {
            rejected_empty_front = true;
        }
        assert(rejected_empty_front);
        bool rejected_empty_back = false;
        try
        {
            (void)values.back();
        }
        catch (const std::out_of_range &)
        {
            rejected_empty_back = true;
        }
        assert(rejected_empty_back);
        bool rejected_empty_pop = false;
        try
        {
            values.pop_back();
        }
        catch (const std::out_of_range &)
        {
            rejected_empty_pop = true;
        }
        assert(rejected_empty_pop);
        values.push_back(1);
        assert(static_cast<int>(values.at(0)) == 1);
        values[0] = 2;
        assert(static_cast<int>(values.front()) == 2);
        assert(static_cast<int>(values.back()) == 2);
        assert(static_cast<int>(values.emplaceBack(3)) == 3);

        {
            auto it = values.begin();
            assert(*it == 2);
            std::atomic<bool> writer_started{false};
            std::atomic<bool> writer_finished{false};
            std::thread writer([&]() {
                writer_started.store(true, std::memory_order_release);
                values.push_back(4);
                writer_finished.store(true, std::memory_order_release);
            });
            while (!writer_started.load(std::memory_order_acquire))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            assert(!writer_finished.load(std::memory_order_acquire));
            it = {};
            writer.join();
            assert(writer_finished.load(std::memory_order_acquire));
        }
        assert(values.size() == 3);

        const auto copied = values.copy();
        assert(copied.size() == 3);

        {
            nerve::threading::ThreadPool pool(1);
            std::atomic<bool> first_started{false};
            std::atomic<bool> release_first{false};
            std::atomic<bool> second_completed{false};

            auto first = pool.submitToQueue(0, [&]() {
                first_started.store(true, std::memory_order_release);
                while (!release_first.load(std::memory_order_acquire))
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });

            while (!first_started.load(std::memory_order_acquire))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            auto second = pool.submitToQueue(
                0, [&]() { second_completed.store(true, std::memory_order_release); });

            assert(!pool.waitForAllFor(std::chrono::milliseconds(10)));
            release_first.store(true, std::memory_order_release);
            assert(pool.waitForAllFor(std::chrono::seconds(2)));
            first.get();
            second.get();
            assert(second_completed.load(std::memory_order_acquire));
            assert(pool.allTasksCompleted());

            const auto stats = pool.getStats();
            assert(stats.total_tasks == 2);
            assert(stats.completed_tasks == 2);
            assert(stats.failed_tasks == 0);
            assert(stats.completion_rate == 1.0);
        }

        {
            nerve::threading::ThreadPool pool(1);
            auto failing = pool.submit([]() -> int { throw std::runtime_error("expected"); });

            assert(pool.waitForAllFor(std::chrono::seconds(2)));

            bool observed_exception = false;
            try
            {
                (void)failing.get();
            }
            catch (const std::runtime_error &)
            {
                observed_exception = true;
            }
            assert(observed_exception);
            assert(pool.allTasksCompleted());

            const auto stats = pool.getStats();
            assert(stats.total_tasks == 1);
            assert(stats.completed_tasks == 0);
            assert(stats.failed_tasks == 1);
            assert(stats.completion_rate == 1.0);
        }

        {
            nerve::threading::ThreadPool pool(1);
            std::atomic<bool> first_started{false};
            std::atomic<bool> release_first{false};
            std::atomic<bool> second_completed{false};

            auto first = pool.submitToQueue(0, [&]() {
                first_started.store(true, std::memory_order_release);
                while (!release_first.load(std::memory_order_acquire))
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });

            while (!first_started.load(std::memory_order_acquire))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            auto second = pool.submitToQueue(
                0, [&]() { second_completed.store(true, std::memory_order_release); });

            pool.shutdown();

            bool submit_rejected = false;
            try
            {
                (void)pool.submit([] {});
            }
            catch (const std::runtime_error &)
            {
                submit_rejected = true;
            }
            assert(submit_rejected);

            release_first.store(true, std::memory_order_release);
            assert(pool.waitForAllFor(std::chrono::seconds(2)));
            first.get();
            second.get();
            assert(second_completed.load(std::memory_order_acquire));

            const auto stats = pool.getStats();
            assert(stats.total_tasks == 2);
            assert(stats.completed_tasks == 2);
            assert(stats.failed_tasks == 0);
            assert(stats.completion_rate == 1.0);
        }

        return 0;
    }
    catch (const std::exception &)
    {
        return 1;
    }
}
