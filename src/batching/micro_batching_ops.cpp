#include "nerve/batching/micro_batching.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace nerve::batching
{

struct MicroBatchProcessor::Impl
{
    Size batch_size = 32;
    Size max_pending = 128;
    std::vector<std::vector<double>> pending_inputs;
    std::vector<std::vector<double>> pending_outputs;
    std::atomic<Size> processed{0};
    std::atomic<Size> batches_completed{0};
};

MicroBatchProcessor::MicroBatchProcessor(Size batch_size, Size max_pending)
    : impl_(std::make_unique<Impl>())
{
    impl_->batch_size = batch_size;
    impl_->max_pending = max_pending;
    impl_->pending_inputs.reserve(max_pending);
    impl_->pending_outputs.reserve(max_pending);
}

MicroBatchProcessor::~MicroBatchProcessor() = default;

void MicroBatchProcessor::submit(const std::vector<double> &input)
{
    impl_->pending_inputs.push_back(input);
    if (impl_->pending_inputs.size() >= impl_->batch_size)
        flushBatch();
}

void MicroBatchProcessor::flushBatch()
{
    if (impl_->pending_inputs.empty())
        return;
    impl_->pending_outputs.resize(impl_->pending_inputs.size());
    for (Size i = 0; i < impl_->pending_inputs.size(); ++i)
        impl_->pending_outputs[i] = processSingle(impl_->pending_inputs[i]);
    impl_->processed += impl_->pending_inputs.size();
    impl_->batches_completed++;
    impl_->pending_inputs.clear();
}

std::vector<double> MicroBatchProcessor::processSingle(const std::vector<double> &input) const
{
    std::vector<double> output(input.size());
    for (Size i = 0; i < input.size(); ++i)
        output[i] = std::tanh(input[i]);
    return output;
}

Size MicroBatchProcessor::processed() const
{
    return impl_->processed.load(std::memory_order_relaxed);
}

Size MicroBatchProcessor::batchesCompleted() const
{
    return impl_->batches_completed.load(std::memory_order_relaxed);
}

Size MicroBatchProcessor::pending() const
{
    return impl_->pending_inputs.size();
}

void MicroBatchProcessor::setBatchSize(Size batch_size)
{
    impl_->batch_size = batch_size;
}

} // namespace nerve::batching
