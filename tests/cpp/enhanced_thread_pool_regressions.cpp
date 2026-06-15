#include "nerve/core/memory/thread_safe_memory_pool.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <exception>
#include <future>
#include <stdexcept>
#include <thread>

int main()
{
    try
    {
        {
            nerve::core::EnhancedThreadPool pool(1);
            assert(pool.numThreads() == 1);

            auto value = pool.submit([] { return 42; });
            pool.waitForAll();
            assert(value.get() == 42);
        }

        {
            nerve::core::EnhancedThreadPool pool(1);
            auto failing = pool.submit([]() -> int { throw std::runtime_error("expected"); });
            pool.waitForAll();

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
        }

        {
            std::atomic<bool> first_started{false};
            std::atomic<bool> release_first{false};
            std::atomic<bool> second_completed{false};
            std::future<void> first;
            std::future<void> second;

            {
                nerve::core::EnhancedThreadPool pool(1);
                first = pool.submit([&] {
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

                second =
                    pool.submit([&] { second_completed.store(true, std::memory_order_release); });
                release_first.store(true, std::memory_order_release);
            }

            first.get();
            second.get();
            assert(second_completed.load(std::memory_order_acquire));
        }

        return 0;
    }
    catch (const std::exception &)
    {
        return 1;
    }
}
