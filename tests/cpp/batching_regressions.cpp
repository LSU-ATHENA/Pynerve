#include "nerve/batching/micro_batching.hpp"
#include "nerve/common/accelerated_types.hpp"
#include "nerve/core/policy/ownership_policy.hpp"
#include "nerve/core_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace
{

using nerve::core::BufferView;

constexpr double kTol = 1e-10;

bool check_micro_batch_processor_construction()
{
    nerve::batching::MicroBatchProcessor proc(16, 100);

    if (proc.pending() != 0)
    {
        std::cerr << "fresh processor should have 0 pending\n";
        return false;
    }
    if (proc.processed() != 0)
    {
        std::cerr << "fresh processor should have 0 processed\n";
        return false;
    }
    if (proc.batchesCompleted() != 0)
    {
        std::cerr << "fresh processor should have 0 batches\n";
        return false;
    }

    return true;
}

bool check_micro_batch_submit_and_flush()
{
    nerve::batching::MicroBatchProcessor proc(32, 200);

    std::vector<double> input(16, 1.0);
    proc.submit(input);

    if (proc.pending() == 0 && proc.processed() == 0)
    {
        proc.flushBatch();
    }

    if (proc.pending() > 200)
    {
        std::cerr << "pending exceeded max\n";
        return false;
    }

    return true;
}

bool check_batch_config_defaults()
{
    nerve::batching::BatchConfig cfg;

    if (cfg.max_batch_size == 0)
    {
        std::cerr << "max_batch_size should be > 0\n";
        return false;
    }
    if (cfg.min_batch_size == 0 || cfg.min_batch_size > cfg.max_batch_size)
    {
        std::cerr << "invalid min_batch_size\n";
        return false;
    }
    if (cfg.num_batch_threads == 0)
    {
        std::cerr << "need at least one batch thread\n";
        return false;
    }

    return true;
}

bool check_batch_item_validity()
{
    nerve::batching::BatchItem<std::vector<double>> item;
    item.data = {1.0, 2.0, 3.0};
    item.timestamp_ns = 1000;
    item.symbol_id = 42;
    item.priority = 0;
    item.sequence_id = 1;

    if (!item.isValid())
    {
        std::cerr << "valid batch item reported invalid\n";
        return false;
    }

    nerve::batching::BatchItem<std::vector<double>> invalid_item;
    invalid_item.data = {};
    invalid_item.timestamp_ns = 0;
    invalid_item.symbol_id = -1;

    if (invalid_item.isValid())
    {
        std::cerr << "invalid batch item reported valid\n";
        return false;
    }

    return true;
}

bool check_micro_batching_engine_construction()
{
    nerve::batching::BatchConfig cfg;
    cfg.max_batch_size = 16;
    cfg.min_batch_size = 1;
    cfg.num_batch_threads = 1;

    nerve::batching::MicroBatchingEngine<std::vector<double>> engine(cfg);
    auto stats = engine.getStats();

    if (stats.total_items_processed != 0)
    {
        std::cerr << "fresh engine should have 0 items processed\n";
        return false;
    }
    if (stats.total_batches_processed != 0)
    {
        std::cerr << "fresh engine should have 0 batches\n";
        return false;
    }

    return true;
}

bool check_symbol_batching_config()
{
    nerve::batching::SymbolBatchingEngine<std::vector<double>>::SymbolBatchConfig cfg;
    cfg.symbol_ids = {1, 2, 3};
    cfg.max_batch_size_per_symbol = 8;
    cfg.max_total_batch_size = 32;

    if (cfg.symbol_ids.empty())
    {
        std::cerr << "symbol IDs should not be empty\n";
        return false;
    }
    if (cfg.max_batch_size_per_symbol == 0)
    {
        std::cerr << "max batch per symbol should be > 0\n";
        return false;
    }

    nerve::batching::SymbolBatchingEngine<std::vector<double>> engine(cfg);
    static_cast<void>(engine);

    return true;
}

} // namespace

int main()
{
    if (!check_micro_batch_processor_construction())
    {
        std::cerr << "FAIL: micro batch processor construction\n";
        return 1;
    }
    if (!check_micro_batch_submit_and_flush())
    {
        std::cerr << "FAIL: micro batch submit and flush\n";
        return 1;
    }
    if (!check_batch_config_defaults())
    {
        std::cerr << "FAIL: batch config defaults\n";
        return 1;
    }
    if (!check_batch_item_validity())
    {
        std::cerr << "FAIL: batch item validity\n";
        return 1;
    }
    if (!check_micro_batching_engine_construction())
    {
        std::cerr << "FAIL: micro batching engine construction\n";
        return 1;
    }
    if (!check_symbol_batching_config())
    {
        std::cerr << "FAIL: symbol batching config\n";
        return 1;
    }
    return 0;
}
