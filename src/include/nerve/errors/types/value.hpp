
#pragma once

#include "nerve/errors/detail/base.hpp"

#include <optional>
#include <vector>

namespace nerve::errors
{

class InvalidArgumentError : public NerveError
{
public:
    InvalidArgumentError()
        : NerveError("Invalid argument")
    {}

    InvalidArgumentError(const char *file, int line, const char *function)
        : NerveError("Invalid argument", file, line, function)
    {}

    InvalidArgumentError &setArgumentName(const std::string &name)
    {
        addContext("argument", name);
        return *this;
    }

    template <typename T>
    InvalidArgumentError &setArgumentValue(const T &value)
    {
        addContext("value", value);
        return *this;
    }

    template <typename T>
    InvalidArgumentError &setExpectedRange(const T &min, const T &max)
    {
        addContext("expected_range", "[" + std::to_string(min) + ", " + std::to_string(max) + "]");
        return *this;
    }

    InvalidArgumentError &setReason(const std::string &reason)
    {
        addContext("reason", reason);
        return *this;
    }

    std::string errorTypeName() const override { return "InvalidArgumentError"; }

    uint32_t errorCode() const override { return 0x00000604; }
};

class BudgetExceededError : public NerveError
{
public:
    BudgetExceededError()
        : NerveError("Budget exceeded")
    {}

    BudgetExceededError(const char *file, int line, const char *function)
        : NerveError("Budget exceeded", file, line, function)
    {}

    BudgetExceededError &setBudgetType(const std::string &type)
    {
        addContext("budget_type", type);
        return *this;
    }

    BudgetExceededError &setBudgetLimit(double limit)
    {
        addContext("budget_limit", std::to_string(limit));
        return *this;
    }

    BudgetExceededError &setActualUsage(double usage)
    {
        addContext("actual_usage", std::to_string(usage));
        return *this;
    }

    BudgetExceededError &setTimeBudgetMs(double ms)
    {
        addContext("time_budget_ms", std::to_string(ms));
        return *this;
    }

    BudgetExceededError &setTimeUsedMs(double ms)
    {
        addContext("time_used_ms", std::to_string(ms));
        return *this;
    }

    BudgetExceededError &setMemoryBudgetMb(double mb)
    {
        addContext("memory_budget_mb", std::to_string(mb));
        return *this;
    }

    BudgetExceededError &setMemoryUsedMb(double mb)
    {
        addContext("memory_used_mb", std::to_string(mb));
        return *this;
    }

    std::string errorTypeName() const override { return "BudgetExceededError"; }

    uint32_t errorCode() const override { return 0x00000603; }
};

class IOError : public NerveError
{
public:
    IOError()
        : NerveError("IO operation failed")
    {}

    IOError(const char *file, int line, const char *function)
        : NerveError("IO operation failed", file, line, function)
    {}

    IOError &setFilePath(const std::string &path)
    {
        addContext("file", path);
        return *this;
    }

    IOError &setIoOperation(const std::string &op)
    {
        addContext("operation", op);
        return *this;
    }

    IOError &setErrno(int err)
    {
        addContext("errno", std::to_string(err));
        return *this;
    }

    std::string errorTypeName() const override { return "IOError"; }

    uint32_t errorCode() const override { return 0x00000100; }
};

class DeterminismError : public NerveError
{
public:
    DeterminismError()
        : NerveError("Determinism violation")
    {}

    DeterminismError(const char *file, int line, const char *function)
        : NerveError("Determinism violation", file, line, function)
    {}

    DeterminismError &setExpectedHash(const std::string &hash)
    {
        addContext("expected_hash", hash);
        return *this;
    }

    DeterminismError &setActualHash(const std::string &hash)
    {
        addContext("actual_hash", hash);
        return *this;
    }

    DeterminismError &setSeed(uint64_t seed)
    {
        addContext("seed", std::to_string(seed));
        return *this;
    }

    std::string errorTypeName() const override { return "DeterminismError"; }

    uint32_t errorCode() const override { return 0x00000400; }
};

class NUMAError : public NerveError
{
public:
    NUMAError()
        : NerveError("NUMA operation failed")
    {}

    NUMAError(const char *file, int line, const char *function)
        : NerveError("NUMA operation failed", file, line, function)
    {}

    NUMAError &setNodeId(int node)
    {
        addContext("numa_node", std::to_string(node));
        return *this;
    }

    NUMAError &setNumaOperation(const std::string &op)
    {
        addContext("numa_operation", op);
        return *this;
    }

    std::string errorTypeName() const override { return "NUMAError"; }

    uint32_t errorCode() const override { return 0x00000700; }
};

class PersistenceError : public NerveError
{
public:
    PersistenceError()
        : NerveError("Persistence computation failed")
    {}

    PersistenceError(const char *file, int line, const char *function)
        : NerveError("Persistence computation failed", file, line, function)
    {}

    PersistenceError &setAlgorithm(const std::string &algo)
    {
        addContext("algorithm", algo);
        return *this;
    }

    PersistenceError &setComplexSize(size_t size)
    {
        addContext("complex_size", std::to_string(size));
        return *this;
    }

    PersistenceError &setMaxDimension(int dim)
    {
        addContext("max_dimension", std::to_string(dim));
        return *this;
    }

    PersistenceError &setNumPoints(size_t n)
    {
        addContext("num_points", std::to_string(n));
        return *this;
    }

    std::string errorTypeName() const override { return "PersistenceError"; }

    uint32_t errorCode() const override { return 0x00000600; }
};

class BettiError : public NerveError
{
public:
    BettiError()
        : NerveError("Betti number computation failed")
    {}

    BettiError(const char *file, int line, const char *function)
        : NerveError("Betti number computation failed", file, line, function)
    {}

    BettiError &setExpectedBetti(const std::vector<int> &betti)
    {
        addContext("expected_betti", formatShape(betti));
        return *this;
    }

    BettiError &setActualBetti(const std::vector<int> &betti)
    {
        addContext("actual_betti", formatShape(betti));
        return *this;
    }

    BettiError &setDimension(int dim)
    {
        addContext("dimension", std::to_string(dim));
        return *this;
    }

    std::string errorTypeName() const override { return "BettiError"; }

    uint32_t errorCode() const override { return 0x00000906; }
};

} // namespace nerve::errors
