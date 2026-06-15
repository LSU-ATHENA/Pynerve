
#pragma once

#include "nerve/algebra/complex.hpp"
#include "nerve/core/budget.hpp"
#include "nerve/core/compact_summary/compact_summary.hpp"
#include "nerve/core_types.hpp"
#include "nerve/errors/errors.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nerve
{

using SimplicialComplex = algebra::SimplicialComplex;
using PointCloud = std::vector<std::vector<double>>;
using CompactSummary = core::CompactSummary;

enum class CohomologyOrdering
{
    STANDARD = 0,
    CLEVER = 1
};

enum class WitnessSampling
{
    RANDOM = 0,
    HIERARCHICAL = 1
};

enum class IncrementalStrategy
{
    RECOMPUTE = 0,
    LOCAL_UPDATE = 1
};

struct PersistencePairRecord
{
    double birth = 0.0;
    double death = 0.0;
    std::size_t dimension = 0;
};

class Diagram
{
public:
    Diagram() = default;

    void addPair(double birth, double death, std::size_t dimension)
    {
        pairs_.push_back(PersistencePairRecord{birth, death, dimension});
    }

    std::size_t size() const { return pairs_.size(); }

    bool empty() const { return pairs_.empty(); }

    void clear() { pairs_.clear(); }

    const std::vector<PersistencePairRecord> &pairs() const { return pairs_; }

private:
    std::vector<PersistencePairRecord> pairs_;
};

template <typename T>
class ErrorResult
{
public:
    static ErrorResult success(const T &value) { return ErrorResult(value); }

    static ErrorResult success(T &&value) { return ErrorResult(std::move(value)); }

    static ErrorResult error(ErrorCode code, const std::string &message)
    {
        return ErrorResult(code, message);
    }

    bool ok() const { return success_; }

    const T &value() const
    {
        if (!success_)
        {
            throw std::runtime_error("attempted to read value from failed ErrorResult");
        }
        return value_;
    }

    T &value()
    {
        if (!success_)
        {
            throw std::runtime_error("attempted to read value from failed ErrorResult");
        }
        return value_;
    }

    ErrorCode errorCode() const { return error_code_; }

    const std::string &errorMessage() const { return error_message_; }

private:
    explicit ErrorResult(const T &value)
        : value_(value)
        , success_(true)
        , error_code_(ErrorCode::SUCCESS)
    {}

    explicit ErrorResult(T &&value)
        : value_(std::move(value))
        , success_(true)
        , error_code_(ErrorCode::SUCCESS)
    {}

    ErrorResult(ErrorCode code, std::string message)
        : value_()
        , success_(false)
        , error_code_(code)
        , error_message_(std::move(message))
    {}

    T value_{};
    bool success_ = false;
    ErrorCode error_code_ = ErrorCode::SUCCESS;
    std::string error_message_;
};

using ResultType = ErrorResult<Diagram>;

struct IncrementalOperation
{
    enum class Type
    {
        ADD_POINT,
        REMOVE_POINT,
        ADD_BATCH,
        REMOVE_BATCH
    };

    Type type = Type::ADD_POINT;
    std::size_t point_index = 0;
    std::vector<double> point;
    PointCloud batch_points;

    std::size_t size() const
    {
        if (!batch_points.empty())
        {
            return batch_points.size();
        }
        return point.empty() ? 0 : 1;
    }
};

class StreamWindow
{
public:
    StreamWindow() = default;

    explicit StreamWindow(PointCloud points)
        : points_(std::move(points))
    {}

    std::size_t size() const { return points_.size(); }

    const PointCloud &points() const { return points_; }

    void push_back(const std::vector<double> &point) { points_.push_back(point); }

private:
    PointCloud points_;
};

} // namespace nerve
