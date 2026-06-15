#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "nerve/core_types.hpp"

namespace nerve::persistence::detail {

template <typename T>
struct ReductionColumn {
    std::vector<T> entries;
    std::vector<Index> row_indices;
    Index pivot = -1;
    bool empty() const { return entries.empty(); }
    void clear() { entries.clear(); row_indices.clear(); pivot = -1; }
    Size size() const { return entries.size(); }
};

template <typename T>
T readColumnEntry(const ReductionColumn<T>& col, Index row) {
    for (Size i = 0; i < col.entries.size(); ++i) {
        if (col.row_indices[i] == row) return col.entries[i];
    }
    return T{0};
}

template <typename T>
void addColumnEntry(ReductionColumn<T>& col, Index row, T value) {
    for (Size i = 0; i < col.entries.size(); ++i) {
        if (col.row_indices[i] == row) {
            col.entries[i] += value;
            if (std::abs(col.entries[i]) < std::numeric_limits<T>::epsilon()) {
                col.entries[i] = T{0};
            }
            return;
        }
    }
    col.entries.push_back(value);
    col.row_indices.push_back(row);
}

template <typename T>
void compressColumn(ReductionColumn<T>& col) {
    Size write = 0;
    for (Size read = 0; read < col.entries.size(); ++read) {
        if (std::abs(col.entries[read]) > std::numeric_limits<T>::epsilon()) {
            if (write != read) {
                col.entries[write] = col.entries[read];
                col.row_indices[write] = col.row_indices[read];
            }
            ++write;
        }
    }
    col.entries.resize(write);
    col.row_indices.resize(write);
}

template <typename T>
void addToColumn(ReductionColumn<T>& target, const ReductionColumn<T>& source, T scalar) {
    for (Size i = 0; i < source.entries.size(); ++i) {
        T val = source.entries[i] * scalar;
        addColumnEntry(target, source.row_indices[i], val);
    }
}

template <typename T>
Index findPivot(const ReductionColumn<T>& col) {
    Index pivot = -1;
    for (Size i = 0; i < col.entries.size(); ++i) {
        if (std::abs(col.entries[i]) > std::numeric_limits<T>::epsilon()) {
            pivot = std::max(pivot, col.row_indices[i]);
        }
    }
    return pivot;
}

template <typename T>
struct StandardReductionEngine {
    std::vector<ReductionColumn<T>>& columns;
    std::vector<Index>& pivot_to_column;
    std::vector<Pair>& pairs;
    std::vector<T>& filtration_values;

    explicit StandardReductionEngine(std::vector<ReductionColumn<T>>& cols,
                                      std::vector<Index>& pivots,
                                      std::vector<Pair>& results,
                                      std::vector<T>& filts)
        : columns(cols), pivot_to_column(pivots), pairs(results),
          filtration_values(filts) {}

    void reduce(Index j) {
        ReductionColumn<T>& col = columns[j];
        while (true) {
            Index pivot = findPivot<T>(col);
            if (pivot < 0) break;
            if (pivot_to_column[pivot] < 0) {
                pivot_to_column[pivot] = j;
                pairs.push_back(Pair{0, filtration_values[pivot], filtration_values[j]});
                break;
            }
            addToColumn<T>(col, columns[pivot_to_column[pivot]], T{-1});
        }
    }

    void run() {
        for (Index j = 0; j < static_cast<Index>(columns.size()); ++j) {
            reduce(j);
            compressColumn<T>(columns[j]);
        }
    }
};

template <typename T>
struct ClearingReductionEngine {
    StandardReductionEngine<T> engine;

    explicit ClearingReductionEngine(std::vector<ReductionColumn<T>>& cols,
                                      std::vector<Index>& pivots,
                                      std::vector<Pair>& results,
                                      std::vector<T>& filts)
        : engine(cols, pivots, results, filts) {}

    void reduce(Index j) {
        engine.reduce(j);
        Index pivot = findPivot<T>(engine.columns[j]);
        if (pivot >= 0) {
            Index birth_column = engine.pivot_to_column[pivot];
            if (birth_column >= 0 && birth_column != j) {
                engine.columns[birth_column].clear();
            }
        }
    }

    void run() {
        for (Index j = static_cast<Index>(engine.columns.size()) - 1; j >= 0; --j) {
            reduce(j);
            compressColumn<T>(engine.columns[j]);
        }
    }
};

}  // namespace nerve::persistence::detail
