
#include "nerve/instrumentation/error_events.hpp"

#include <iostream>

namespace nerve::instrumentation
{

class HighDimErrorSink
{
public:
    explicit HighDimErrorSink(bool enable_logging = false)
        : logging_enabled_(enable_logging)
    {}

    void handleErrorEvent(const ErrorEvent &event)
    {
        ++total_events_;
        if (event.max_dimension_attempted >= 4)
        {
            ++high_dim_events_;
        }
        if (event.truncated_by_budget)
        {
            ++budget_truncations_;
        }
        if (!logging_enabled_)
        {
            return;
        }
        std::cerr << "[high_dim_error_sink]"
                  << " code=" << static_cast<int>(event.code) << " ts_ns=" << event.timestamp_ns
                  << " max_dim=" << event.max_dimension_attempted
                  << " boundary_ops=" << event.num_boundary_ops
                  << " truncated=" << (event.truncated_by_budget ? 1 : 0)
                  << " precision=" << static_cast<int>(event.precision_level) << '\n';
    }

    std::size_t totalEvents() const { return total_events_; }
    std::size_t highDimEvents() const { return high_dim_events_; }
    std::size_t budgetTruncations() const { return budget_truncations_; }

private:
    bool logging_enabled_ = false;
    std::size_t total_events_ = 0;
    std::size_t high_dim_events_ = 0;
    std::size_t budget_truncations_ = 0;
};

} // namespace nerve::instrumentation
