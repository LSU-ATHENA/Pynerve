
#pragma once

#include "nerve/errors/detail/base.hpp"

#include <optional>
#include <vector>

namespace nerve::errors
{

class ShapeMismatchError : public NerveError
{
public:
    ShapeMismatchError()
        : NerveError("Shape mismatch")
    {}

    ShapeMismatchError(const char *file, int line, const char *function)
        : NerveError("Shape mismatch", file, line, function)
    {}

    ShapeMismatchError &setExpectedShape(const std::vector<size_t> &shape)
    {
        expected_shape_ = shape;
        addContext("expected_shape", formatShape(shape));
        return *this;
    }

    ShapeMismatchError &setActualShape(const std::vector<size_t> &shape)
    {
        actual_shape_ = shape;
        addContext("actual_shape", formatShape(shape));
        return *this;
    }

    ShapeMismatchError &setExpectedDims(size_t rows, size_t cols)
    {
        expected_shape_ = {rows, cols};
        addContext("expected_shape",
                   "[" + std::to_string(rows) + ", " + std::to_string(cols) + "]");
        return *this;
    }

    ShapeMismatchError &setActualAccess(size_t row, size_t col)
    {
        addContext("attempted_access",
                   "[" + std::to_string(row) + ", " + std::to_string(col) + "]");
        return *this;
    }

    ShapeMismatchError &setDimensionMismatch(size_t dim)
    {
        mismatched_dimension_ = dim;
        addContext("mismatched_dimension", std::to_string(dim));

        if (expected_shape_ && actual_shape_ && dim < expected_shape_->size() &&
            dim < actual_shape_->size())
        {
            std::ostringstream oss;
            oss << "Dimension " << dim << ": expected size " << (*expected_shape_)[dim] << ", got "
                << (*actual_shape_)[dim];
            addContext("dimension_detail", oss.str());
        }
        return *this;
    }

    ShapeMismatchError &setMessage(const std::string &msg)
    {
        setBaseMessage(msg);
        return *this;
    }

    std::string errorTypeName() const override { return "ShapeMismatchError"; }

    uint32_t errorCode() const override { return 0x00000900; }

private:
    std::optional<std::vector<size_t>> expected_shape_;
    std::optional<std::vector<size_t>> actual_shape_;
    std::optional<size_t> mismatched_dimension_;
};

class DimensionError : public NerveError
{
public:
    DimensionError()
        : NerveError("Invalid dimension")
    {}

    DimensionError(const char *file, int line, const char *function)
        : NerveError("Invalid dimension", file, line, function)
    {}

    DimensionError &expected(int dim)
    {
        expected_dim_ = dim;
        addContext("expected_dimension", std::to_string(dim));
        return *this;
    }

    DimensionError &actual(int dim)
    {
        actual_dim_ = dim;
        addContext("actual_dimension", std::to_string(dim));
        return *this;
    }

    DimensionError &expected(size_t dim)
    {
        expected_dim_ = static_cast<int>(dim);
        addContext("expected_dimension", std::to_string(dim));
        return *this;
    }

    DimensionError &actual(size_t dim)
    {
        actual_dim_ = static_cast<int>(dim);
        addContext("actual_dimension", std::to_string(dim));
        return *this;
    }

    DimensionError &setSimplexId(size_t id)
    {
        addContext("simplex_id", std::to_string(id));
        return *this;
    }

    DimensionError &setVertices(const std::vector<size_t> &vertices)
    {
        addContext("vertices", formatShape(vertices));
        return *this;
    }

    DimensionError &setMessage(const std::string &msg)
    {
        setBaseMessage(msg);
        return *this;
    }

    std::string errorTypeName() const override { return "DimensionError"; }

    uint32_t errorCode() const override { return 0x00000907; }

private:
    std::optional<int> expected_dim_;
    std::optional<int> actual_dim_;
};

class TypeError : public NerveError
{
public:
    TypeError()
        : NerveError("Type mismatch")
    {}

    TypeError(const char *file, int line, const char *function)
        : NerveError("Type mismatch", file, line, function)
    {}

    TypeError &setExpectedType(const std::string &type)
    {
        addContext("expected_type", type);
        return *this;
    }

    TypeError &setActualType(const std::string &type)
    {
        addContext("actual_type", type);
        return *this;
    }

    TypeError &setMessage(const std::string &msg)
    {
        setBaseMessage(msg);
        return *this;
    }

    std::string errorTypeName() const override { return "TypeError"; }
};

class InvalidSimplexError : public NerveError
{
public:
    InvalidSimplexError()
        : NerveError("Invalid simplex")
    {}

    InvalidSimplexError(const char *file, int line, const char *function)
        : NerveError("Invalid simplex", file, line, function)
    {}

    InvalidSimplexError &setSimplexId(size_t id)
    {
        addContext("simplex_id", std::to_string(id));
        return *this;
    }

    InvalidSimplexError &setVertices(const std::vector<size_t> &vertices)
    {
        addContext("vertices", formatShape(vertices));
        addContext("vertex_count", std::to_string(vertices.size()));
        return *this;
    }

    InvalidSimplexError &setExpectedDim(int dim)
    {
        addContext("expected_dimension", std::to_string(dim));
        return *this;
    }

    InvalidSimplexError &setActualDim(int dim)
    {
        addContext("actual_dimension", std::to_string(dim));
        return *this;
    }

    InvalidSimplexError &setReason(const std::string &reason)
    {
        addContext("reason", reason);
        return *this;
    }

    InvalidSimplexError &setMessage(const std::string &msg)
    {
        setBaseMessage(msg);
        return *this;
    }

    std::string errorTypeName() const override { return "InvalidSimplexError"; }

    uint32_t errorCode() const override { return 0x00000907; }
};

class MatrixStructureError : public NerveError
{
public:
    MatrixStructureError()
        : NerveError("Matrix structure error")
    {}

    MatrixStructureError(const char *file, int line, const char *function)
        : NerveError("Matrix structure error", file, line, function)
    {}

    MatrixStructureError &setRows(size_t rows)
    {
        addContext("rows", std::to_string(rows));
        return *this;
    }

    MatrixStructureError &setCols(size_t cols)
    {
        addContext("cols", std::to_string(cols));
        return *this;
    }

    MatrixStructureError &setNonzeros(size_t nnz)
    {
        addContext("nonzeros", std::to_string(nnz));
        return *this;
    }

    MatrixStructureError &setExpectedDensity(double density)
    {
        addContext("expected_density", std::to_string(density));
        return *this;
    }

    MatrixStructureError &setActualDensity(double density)
    {
        addContext("actual_density", std::to_string(density));
        return *this;
    }

    MatrixStructureError &setReason(const std::string &reason)
    {
        addContext("reason", reason);
        return *this;
    }

    MatrixStructureError &setMessage(const std::string &msg)
    {
        setBaseMessage(msg);
        return *this;
    }

    std::string errorTypeName() const override { return "MatrixStructureError"; }

    uint32_t errorCode() const override { return 0x00000904; }
};

} // namespace nerve::errors
