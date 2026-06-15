// ShardedBoundaryMatrix Implementation

#include "distributed_wire_format.hpp"
#include "nerve/distributed/mpi_persistence.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace nerve::distributed
{
namespace
{

void normalizeColumn(std::vector<int> *column)
{
    if (column == nullptr)
    {
        return;
    }
    std::sort(column->begin(), column->end());
    column->erase(std::unique(column->begin(), column->end()), column->end());
}

void validateColumn(const std::vector<int> &column)
{
    if (std::any_of(column.begin(), column.end(), [](int row) { return row < 0; }))
    {
        throw std::invalid_argument("distributed boundary rows must be non-negative");
    }
}

void checkMpiSuccess(int status, const char *context)
{
    if (status != MPI_SUCCESS)
    {
        throw std::runtime_error(context);
    }
}

std::vector<int> xorColumns(const std::vector<int> &lhs, const std::vector<int> &rhs)
{
    std::vector<int> out;
    out.reserve(lhs.size() + rhs.size());
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < lhs.size() || j < rhs.size())
    {
        if (j >= rhs.size() || (i < lhs.size() && lhs[i] < rhs[j]))
        {
            out.push_back(lhs[i++]);
            continue;
        }
        if (i >= lhs.size() || rhs[j] < lhs[i])
        {
            out.push_back(rhs[j++]);
            continue;
        }
        ++i;
        ++j;
    }
    return out;
}

bool readSize(const std::vector<uint8_t> &data, std::size_t *offset, std::size_t *value)
{
    uint64_t raw = 0;
    if (!detail::readUint64LittleEndian(data, offset, &raw))
    {
        return false;
    }
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t))
    {
        if (raw > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            return false;
        }
    }
    *value = static_cast<std::size_t>(raw);
    return true;
}

bool readInt(const std::vector<uint8_t> &data, std::size_t *offset, int *value)
{
    int32_t raw = 0;
    if (!detail::readInt32LittleEndian(data, offset, &raw))
    {
        return false;
    }
    *value = static_cast<int>(raw);
    return true;
}

std::map<int, std::vector<int>> parseSerializedColumns(const std::vector<uint8_t> &data)
{
    std::map<int, std::vector<int>> parsed;
    std::size_t offset = 0;
    std::size_t column_count = 0;
    if (!readSize(data, &offset, &column_count))
    {
        return parsed;
    }

    for (std::size_t i = 0; i < column_count; ++i)
    {
        int index = 0;
        std::size_t column_size = 0;
        if (!readInt(data, &offset, &index) || !readSize(data, &offset, &column_size))
        {
            break;
        }
        if (column_size > (data.size() - offset) / detail::kUint32WireSize)
        {
            break;
        }
        if (!detail::canRead(data, offset, column_size * detail::kUint32WireSize))
        {
            break;
        }

        std::vector<int> column;
        column.reserve(column_size);
        for (std::size_t row_index = 0; row_index < column_size; ++row_index)
        {
            int row = 0;
            if (!readInt(data, &offset, &row))
            {
                return parsed;
            }
            column.push_back(row);
        }
        validateColumn(column);
        normalizeColumn(&column);
        parsed[index] = std::move(column);
    }
    return parsed;
}

} // namespace

ShardedBoundaryMatrix::ShardedBoundaryMatrix(int world_rank, int world_size)
    : rank_(world_rank)
    , size_(std::max(1, world_size))
    , num_columns_(0)
{}

void ShardedBoundaryMatrix::distribute_columns(const std::vector<std::vector<int>> &columns)
{
    if (columns.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("distributed boundary column count exceeds int range");
    }
    for (const auto &column : columns)
    {
        validateColumn(column);
    }
    num_columns_ = columns.size();
    local_columns_.clear();
    remote_cache_.clear();
    column_to_rank_.clear();

    for (std::size_t index = 0; index < columns.size(); ++index)
    {
        const int owner = static_cast<int>(index % static_cast<std::size_t>(size_));
        column_to_rank_[static_cast<int>(index)] = owner;
        if (owner != rank_)
        {
            continue;
        }
        std::vector<int> normalized = columns[index];
        normalizeColumn(&normalized);
        local_columns_[static_cast<int>(index)] = std::move(normalized);
    }
}

std::vector<int> ShardedBoundaryMatrix::get_boundary(int simplex_idx)
{
    if (simplex_idx < 0)
    {
        throw std::invalid_argument("distributed boundary simplex index must be non-negative");
    }
    const auto owner_it = column_to_rank_.find(simplex_idx);
    if (owner_it == column_to_rank_.end())
    {
        return {};
    }
    const int owner = owner_it->second;
    if (owner == rank_)
    {
        const auto local_it = local_columns_.find(simplex_idx);
        return local_it == local_columns_.end() ? std::vector<int>{} : local_it->second;
    }
    return fetch_remote_boundary(simplex_idx, owner);
}

void ShardedBoundaryMatrix::distributed_reduce()
{
    reduce_local();
    synchronize_globals();
    resolve_remote_dependencies();
}

void ShardedBoundaryMatrix::checkpoint(const std::string &path)
{
    const std::vector<uint8_t> data = serialize();
    if (rank_ == 0)
    {
        write_checkpoint_metadata(path, num_columns_);
    }
    checkMpiSuccess(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier failed during checkpoint");
    write_checkpoint_data(path + ".rank" + std::to_string(rank_), data);
}

void ShardedBoundaryMatrix::restore(const std::string &path)
{
    const std::vector<uint8_t> data = read_checkpoint_data(path + ".rank" + std::to_string(rank_));
    deserialize(data);
    checkMpiSuccess(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier failed during restore");
}

std::vector<int> ShardedBoundaryMatrix::fetch_remote_boundary(int idx, int owner)
{
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        const auto cached = remote_cache_.find(idx);
        if (cached != remote_cache_.end())
        {
            return cached->second;
        }
    }

    if (owner == rank_)
    {
        const auto local_it = local_columns_.find(idx);
        return local_it == local_columns_.end() ? std::vector<int>{} : local_it->second;
    }
    return {};
}

void ShardedBoundaryMatrix::reduce_local()
{
    std::vector<int> ordered_columns;
    ordered_columns.reserve(local_columns_.size());
    for (const auto &[index, _] : local_columns_)
    {
        ordered_columns.push_back(index);
    }
    std::sort(ordered_columns.begin(), ordered_columns.end());

    std::unordered_map<int, int> pivot_to_column;
    for (const int column_index : ordered_columns)
    {
        auto &column = local_columns_[column_index];
        normalizeColumn(&column);

        while (!column.empty())
        {
            const int pivot = column.back();
            const auto pivot_it = pivot_to_column.find(pivot);
            if (pivot_it == pivot_to_column.end())
            {
                break;
            }
            const auto reducer_it = local_columns_.find(pivot_it->second);
            if (reducer_it == local_columns_.end())
            {
                break;
            }
            column = xorColumns(column, reducer_it->second);
        }

        if (!column.empty())
        {
            pivot_to_column[column.back()] = column_index;
        }
    }
}

void ShardedBoundaryMatrix::synchronize_globals()
{
    const std::vector<uint8_t> local_payload = serialize();
    const int local_size = detail::checkedMpiByteCount(
        local_payload.size(), "distributed boundary payload exceeds MPI byte-count range");

    std::vector<int> recv_sizes(static_cast<std::size_t>(size_), 0);
    checkMpiSuccess(
        MPI_Allgather(&local_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD),
        "MPI_Allgather failed during boundary synchronization");

    std::vector<int> displacements(static_cast<std::size_t>(size_), 0);
    const int total_bytes = detail::buildMpiDisplacements(
        recv_sizes, &displacements, "distributed boundary gather exceeds MPI byte-count range");

    std::vector<uint8_t> gathered(static_cast<std::size_t>(std::max(0, total_bytes)), 0U);
    const uint8_t *local_data = local_payload.empty() ? nullptr : local_payload.data();
    /* const_cast: safe -- MPI_Allgatherv does not modify the send buffer */
    checkMpiSuccess(MPI_Allgatherv(const_cast<uint8_t *>(local_data), local_size, MPI_BYTE,
                                   gathered.empty() ? nullptr : gathered.data(), recv_sizes.data(),
                                   displacements.data(), MPI_BYTE, MPI_COMM_WORLD),
                    "MPI_Allgatherv failed during boundary synchronization");

    std::lock_guard<std::mutex> lock(cache_mutex_);
    remote_cache_.clear();
    for (int rank = 0; rank < size_; ++rank)
    {
        const int size_bytes = recv_sizes[static_cast<std::size_t>(rank)];
        const int displacement = displacements[static_cast<std::size_t>(rank)];
        if (size_bytes <= 0 || displacement < 0 || displacement + size_bytes > total_bytes)
        {
            continue;
        }
        std::vector<uint8_t> payload(gathered.begin() + displacement,
                                     gathered.begin() + displacement + size_bytes);
        const auto parsed = parseSerializedColumns(payload);
        for (const auto &[index, column] : parsed)
        {
            remote_cache_[index] = column;
        }
    }
}

void ShardedBoundaryMatrix::resolve_remote_dependencies()
{
    std::map<int, std::vector<int>> global_columns;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (const auto &[index, column] : remote_cache_)
        {
            global_columns[index] = column;
        }
    }
    for (const auto &[index, column] : local_columns_)
    {
        global_columns[index] = column;
    }

    std::unordered_map<int, int> pivot_to_column;
    for (auto &[column_index, column] : global_columns)
    {
        normalizeColumn(&column);
        while (!column.empty())
        {
            const int pivot = column.back();
            const auto pivot_it = pivot_to_column.find(pivot);
            if (pivot_it == pivot_to_column.end())
            {
                break;
            }
            const auto reducer_it = global_columns.find(pivot_it->second);
            if (reducer_it == global_columns.end())
            {
                break;
            }
            column = xorColumns(column, reducer_it->second);
        }
        if (!column.empty())
        {
            pivot_to_column[column.back()] = column_index;
        }
    }

    local_columns_.clear();
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        remote_cache_.clear();
        for (const auto &[index, column] : global_columns)
        {
            remote_cache_[index] = column;
            const auto owner_it = column_to_rank_.find(index);
            if (owner_it != column_to_rank_.end() && owner_it->second == rank_)
            {
                local_columns_[index] = column;
            }
        }
    }
}

std::vector<uint8_t> ShardedBoundaryMatrix::serialize()
{
    std::vector<std::pair<int, std::vector<int>>> ordered;
    ordered.reserve(local_columns_.size());
    for (const auto &[index, column] : local_columns_)
    {
        ordered.push_back({index, column});
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::vector<uint8_t> result;
    const std::size_t count = ordered.size();
    detail::appendUint64LittleEndian(result, static_cast<std::uint64_t>(count));

    for (const auto &[index, raw_column] : ordered)
    {
        std::vector<int> column = raw_column;
        normalizeColumn(&column);
        const std::size_t column_size = column.size();
        detail::appendInt32LittleEndian(result, static_cast<std::int32_t>(index));
        detail::appendUint64LittleEndian(result, static_cast<std::uint64_t>(column_size));
        for (const int row : column)
        {
            detail::appendInt32LittleEndian(result, static_cast<std::int32_t>(row));
        }
    }
    return result;
}

void ShardedBoundaryMatrix::deserialize(const std::vector<uint8_t> &data)
{
    local_columns_.clear();
    column_to_rank_.clear();
    remote_cache_.clear();
    num_columns_ = 0;
    const auto parsed = parseSerializedColumns(data);
    for (const auto &[index, column] : parsed)
    {
        if (index < 0)
        {
            continue;
        }
        local_columns_[index] = column;
        column_to_rank_[index] = rank_;
        num_columns_ = std::max(num_columns_, static_cast<std::size_t>(index) + 1);
    }
}

void ShardedBoundaryMatrix::write_checkpoint_metadata(const std::string &path, size_t total_cols)
{
    FILE *fp = std::fopen((path + ".meta").c_str(), "wb");
    if (fp == nullptr)
    {
        return;
    }
    std::vector<uint8_t> payload;
    detail::appendUint64LittleEndian(payload, static_cast<std::uint64_t>(total_cols));
    const std::size_t written = std::fwrite(payload.data(), 1, payload.size(), fp);
    if (written != payload.size())
    {
        std::fclose(fp);
        return;
    }
    std::fclose(fp);
}

void ShardedBoundaryMatrix::write_checkpoint_data(const std::string &path,
                                                  const std::vector<uint8_t> &data)
{
    FILE *fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr)
    {
        return;
    }
    if (!data.empty())
    {
        const std::size_t written = std::fwrite(data.data(), 1, data.size(), fp);
        if (written != data.size())
        {
            std::fclose(fp);
            return;
        }
    }
    std::fclose(fp);
}

std::vector<uint8_t> ShardedBoundaryMatrix::read_checkpoint_data(const std::string &path)
{
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr)
    {
        return {};
    }

    if (std::fseek(fp, 0, SEEK_END) != 0)
    {
        std::fclose(fp);
        return {};
    }
    const long raw_size = std::ftell(fp);
    if (raw_size < 0)
    {
        std::fclose(fp);
        return {};
    }
    if (std::fseek(fp, 0, SEEK_SET) != 0)
    {
        std::fclose(fp);
        return {};
    }

    std::vector<uint8_t> data(static_cast<std::size_t>(raw_size), 0U);
    if (!data.empty())
    {
        const std::size_t read = std::fread(data.data(), 1, data.size(), fp);
        data.resize(read);
    }
    std::fclose(fp);
    return data;
}

} // namespace nerve::distributed
