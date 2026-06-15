#include "nerve/sheaf/gpu_sheaf.hpp"

#include <condition_variable>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>

namespace nerve::sheaf::morphism
{

class AsyncMorphismQueue::Impl
{
public:
    ~Impl() { stop(); }

    void submit(int from, int to, const std::vector<float> &input,
                std::promise<std::vector<float>> promise)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_)
        {
            promise.set_exception(
                std::make_exception_ptr(std::runtime_error("AsyncMorphismQueue is stopped")));
            return;
        }
        queue_.push(Task{from, to, input, std::move(promise)});
        cv_.notify_one();
    }

    void start(int workers)
    {
        if (workers < 0)
        {
            throw std::invalid_argument("num_workers must be non-negative");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (running_ || workers == 0)
        {
            return;
        }
        running_ = true;
        for (int i = 0; i < workers; ++i)
        {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            running_ = false;
            if (workers_.empty())
            {
                failQueued("AsyncMorphismQueue stopped before workers started");
            }
        }
        cv_.notify_all();
        for (auto &worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        workers_.clear();
    }

private:
    struct Task
    {
        int from_stalk;
        int to_stalk;
        std::vector<float> input;
        std::promise<std::vector<float>> promise;
    };

    static std::vector<float> compute(const Task &task)
    {
        if (task.from_stalk == task.to_stalk)
        {
            return task.input;
        }
        throw std::invalid_argument(
            "AsyncMorphismQueue requires an explicit morphism for non-identity maps");
    }

    void workerLoop()
    {
        while (true)
        {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
                if (queue_.empty() && !running_)
                {
                    return;
                }
                task = std::move(queue_.front());
                queue_.pop();
            }
            try
            {
                task.promise.set_value(compute(task));
            }
            catch (...)
            {
                task.promise.set_exception(std::current_exception());
            }
        }
    }

    void failQueued(const char *message)
    {
        while (!queue_.empty())
        {
            auto task = std::move(queue_.front());
            queue_.pop();
            task.promise.set_exception(std::make_exception_ptr(std::runtime_error(message)));
        }
    }

    std::queue<Task> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::thread> workers_;
    bool running_ = false;
    bool stopped_ = false;
};

AsyncMorphismQueue::AsyncMorphismQueue()
    : impl_(std::make_unique<Impl>())
{}

AsyncMorphismQueue::~AsyncMorphismQueue() = default;

void AsyncMorphismQueue::submit(int from_stalk, int to_stalk, const std::vector<float> &input,
                                std::promise<std::vector<float>> promise)
{
    impl_->submit(from_stalk, to_stalk, input, std::move(promise));
}

void AsyncMorphismQueue::start(int num_workers)
{
    impl_->start(num_workers);
}

void AsyncMorphismQueue::stop()
{
    impl_->stop();
}

} // namespace nerve::sheaf::morphism
