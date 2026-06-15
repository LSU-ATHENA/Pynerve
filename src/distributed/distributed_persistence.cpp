// DistributedPersistence Implementation

#include "distributed_wire_format.hpp"
#include "nerve/distributed/mpi_persistence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nerve::distributed
{
namespace
{

struct PairRecord
{
    float birth = 0.0f;
    float death = 0.0f;
    int dim = 0;
};

constexpr std::size_t kPairRecordWireSize =
    detail::kUint32WireSize + detail::kUint32WireSize + detail::kUint32WireSize;

double finiteBenchmarkSpeedup(double baseline_ms, double accelerated_ms)
{
    if (!std::isfinite(baseline_ms) || baseline_ms < 0.0 || !std::isfinite(accelerated_ms) ||
        accelerated_ms <= 0.0)
    {
        return 1.0;
    }
    const double speedup = baseline_ms / accelerated_ms;
    return std::isfinite(speedup) && speedup >= 0.0 ? speedup : 1.0;
}

void validatePointClouds(const std::vector<std::vector<float>> &point_clouds)
{
    if (point_clouds.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error("distributed point-cloud count exceeds int range");
    }
    for (const auto &cloud : point_clouds)
    {
        if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::length_error("distributed point-cloud dimension exceeds int range");
        }
        for (const float value : cloud)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("distributed point-cloud values must be finite");
            }
        }
    }
}

void checkMpiSuccess(int status, const char *context)
{
    if (status != MPI_SUCCESS)
    {
        throw std::runtime_error(context);
    }
}

void appendPairRecord(std::vector<std::uint8_t> *output, const PairRecord &record)
{
    detail::appendFloatLittleEndian(*output, record.birth);
    detail::appendFloatLittleEndian(*output, record.death);
    detail::appendInt32LittleEndian(*output, static_cast<std::int32_t>(record.dim));
}

bool readPairRecord(const std::vector<std::uint8_t> &data, std::size_t *offset, PairRecord *record)
{
    std::int32_t dim = 0;
    if (record == nullptr || !detail::readFloatLittleEndian(data, offset, &record->birth) ||
        !detail::readFloatLittleEndian(data, offset, &record->death) ||
        !detail::readInt32LittleEndian(data, offset, &dim))
    {
        return false;
    }
    record->dim = static_cast<int>(dim);
    return true;
}

std::vector<int> buildBoundaryColumn(const std::vector<float> &cloud)
{
    std::vector<int> column;
    if (cloud.empty())
    {
        return column;
    }

    std::vector<float> magnitudes;
    magnitudes.reserve(cloud.size());
    for (const float value : cloud)
    {
        if (std::isfinite(value))
        {
            magnitudes.push_back(std::abs(value));
        }
    }
    if (magnitudes.empty())
    {
        return column;
    }

    std::vector<float> ordered = magnitudes;
    const auto median_it = ordered.begin() + static_cast<std::ptrdiff_t>(ordered.size() / 2);
    std::nth_element(ordered.begin(), median_it, ordered.end());
    const float threshold = std::max(1.0e-8f, *median_it);

    column.reserve(cloud.size() / 2 + 1);
    for (std::size_t index = 0; index < cloud.size(); ++index)
    {
        const float value = cloud[index];
        if (!std::isfinite(value))
        {
            continue;
        }
        if (std::abs(value) >= threshold)
        {
            column.push_back(static_cast<int>(index));
        }
    }

    if (column.empty())
    {
        const auto max_it = std::max_element(cloud.begin(), cloud.end(), [](float lhs, float rhs) {
            return std::abs(lhs) < std::abs(rhs);
        });
        column.push_back(static_cast<int>(std::distance(cloud.begin(), max_it)));
    }

    std::sort(column.begin(), column.end());
    column.erase(std::unique(column.begin(), column.end()), column.end());
    return column;
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

std::vector<std::vector<int>> reduceColumnsSerial(const std::vector<std::vector<int>> &columns)
{
    std::vector<std::vector<int>> reduced = columns;
    std::unordered_map<int, std::size_t> pivot_to_column;

    for (std::size_t col = 0; col < reduced.size(); ++col)
    {
        auto &current = reduced[col];
        std::sort(current.begin(), current.end());
        current.erase(std::unique(current.begin(), current.end()), current.end());

        while (!current.empty())
        {
            const int pivot = current.back();
            const auto pivot_it = pivot_to_column.find(pivot);
            if (pivot_it == pivot_to_column.end())
            {
                break;
            }
            current = xorColumns(current, reduced[pivot_it->second]);
        }

        if (!current.empty())
        {
            pivot_to_column[current.back()] = col;
        }
    }

    return reduced;
}

double measureSingleNodeReductionMs(const std::vector<std::vector<float>> &point_clouds)
{
    const auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> columns;
    columns.reserve(point_clouds.size());
    for (const auto &cloud : point_clouds)
    {
        columns.push_back(buildBoundaryColumn(cloud));
    }
    const auto reduced = reduceColumnsSerial(columns);
    if (reduced.size() != columns.size())
    {
        throw std::runtime_error("serial reduction produced an inconsistent column count");
    }
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

DistributedPersistence::DistributedPersistence()
    : comm_()
    , matrix_(comm_.rank(), comm_.size())
    , scheduler_(comm_.rank(), comm_.size())
    , num_columns_(0)
{}

std::vector<std::tuple<float, float, int>>
DistributedPersistence::compute(const std::vector<std::vector<float>> &point_clouds)
{
    validatePointClouds(point_clouds);
    distribute_filtration(point_clouds);
    comm_.barrier();
    matrix_.distributed_reduce();
    comm_.barrier();
    return gather_results();
}

void DistributedPersistence::distribute_filtration(
    const std::vector<std::vector<float>> &point_clouds)
{
    num_columns_ = point_clouds.size();

    std::vector<std::vector<int>> boundary_columns(num_columns_);
    for (std::size_t column = 0; column < point_clouds.size(); ++column)
    {
        boundary_columns[column] = buildBoundaryColumn(point_clouds[column]);
    }
    matrix_.distribute_columns(boundary_columns);

    // Execute per-rank validation work items so scheduler queues are exercised
    // consistently with the distributed path.
    for (std::size_t column = 0; column < point_clouds.size(); ++column)
    {
        if (static_cast<int>(column % static_cast<std::size_t>(std::max(1, comm_.size()))) !=
            comm_.rank())
        {
            continue;
        }
        scheduler_.submit_work([this, column]() {
            const auto boundary = matrix_.get_boundary(static_cast<int>(column));
            if (boundary.empty())
            {
                return;
            }
            if (!std::is_sorted(boundary.begin(), boundary.end()))
            {
                std::vector<int> sorted = boundary;
                std::sort(sorted.begin(), sorted.end());
                if (sorted != boundary)
                {
                    throw std::runtime_error("distributed boundary column is not sorted");
                }
            }
        });
    }
    scheduler_.run();
}

std::vector<std::tuple<float, float, int>> DistributedPersistence::gather_results()
{
    std::vector<PairRecord> local_pairs;
    local_pairs.reserve(num_columns_ / static_cast<std::size_t>(std::max(1, comm_.size())) + 1);

    for (std::size_t column = 0; column < num_columns_; ++column)
    {
        if (static_cast<int>(column % static_cast<std::size_t>(std::max(1, comm_.size()))) !=
            comm_.rank())
        {
            continue;
        }

        std::vector<int> boundary = matrix_.get_boundary(static_cast<int>(column));
        std::sort(boundary.begin(), boundary.end());
        boundary.erase(std::unique(boundary.begin(), boundary.end()), boundary.end());

        PairRecord record;
        if (boundary.empty())
        {
            record.birth = static_cast<float>(column);
            record.death = std::numeric_limits<float>::infinity();
            record.dim = 0;
        }
        else
        {
            record.birth = static_cast<float>(boundary.back());
            record.death = static_cast<float>(column);
            record.dim = 1;
        }
        local_pairs.push_back(record);
    }

    const int world_size = std::max(1, comm_.size());
    std::vector<std::uint8_t> local_payload;
    local_payload.reserve(local_pairs.size() * kPairRecordWireSize);
    for (const PairRecord &record : local_pairs)
    {
        appendPairRecord(&local_payload, record);
    }
    const int local_bytes = detail::checkedMpiByteCount(
        local_payload.size(), "distributed persistence payload exceeds MPI byte-count range");
    std::vector<int> recv_sizes(static_cast<std::size_t>(world_size), 0);
    checkMpiSuccess(
        MPI_Allgather(&local_bytes, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD),
        "MPI_Allgather failed during persistence result gather");

    std::vector<int> displacements(static_cast<std::size_t>(world_size), 0);
    const int total_bytes = detail::buildMpiDisplacements(
        recv_sizes, &displacements, "distributed persistence gather exceeds MPI byte-count range");

    std::vector<std::uint8_t> packed(static_cast<std::size_t>(std::max(0, total_bytes)), 0U);
    const std::uint8_t *local_data = local_payload.empty() ? nullptr : local_payload.data();
    /* const_cast: safe -- MPI_Allgatherv does not modify the send buffer */
    checkMpiSuccess(MPI_Allgatherv(const_cast<std::uint8_t *>(local_data), local_bytes, MPI_BYTE,
                                   packed.empty() ? nullptr : packed.data(), recv_sizes.data(),
                                   displacements.data(), MPI_BYTE, MPI_COMM_WORLD),
                    "MPI_Allgatherv failed during persistence result gather");

    std::vector<std::tuple<float, float, int>> gathered;
    gathered.reserve(packed.size() / kPairRecordWireSize);
    for (std::size_t offset = 0; offset + kPairRecordWireSize <= packed.size();)
    {
        PairRecord record;
        if (!readPairRecord(packed, &offset, &record))
        {
            break;
        }
        gathered.emplace_back(record.birth, record.death, record.dim);
    }

    std::sort(gathered.begin(), gathered.end(), [](const auto &lhs, const auto &rhs) {
        if (std::get<2>(lhs) != std::get<2>(rhs))
        {
            return std::get<2>(lhs) < std::get<2>(rhs);
        }
        if (std::get<0>(lhs) != std::get<0>(rhs))
        {
            return std::get<0>(lhs) < std::get<0>(rhs);
        }
        return std::get<1>(lhs) < std::get<1>(rhs);
    });
    return gathered;
}

DistributedBenchmark benchmark_distributed(int num_nodes, int data_size_per_node)
{
    DistributedBenchmark bench{};
    bench.num_nodes = num_nodes;
    bench.num_gpus_per_node = 1;
    bench.total_data_size = static_cast<std::size_t>(std::max(0, num_nodes)) *
                            static_cast<std::size_t>(std::max(0, data_size_per_node));

    std::vector<std::vector<float>> test_data(static_cast<std::size_t>(std::max(0, num_nodes)));
    for (int i = 0; i < num_nodes; ++i)
    {
        auto &cloud = test_data[static_cast<std::size_t>(i)];
        cloud.resize(static_cast<std::size_t>(std::max(0, data_size_per_node)));
        for (std::size_t j = 0; j < cloud.size(); ++j)
        {
            cloud[j] = std::sin(static_cast<float>(i + 1) * static_cast<float>(j + 1) * 0.013f);
        }
    }

    DistributedPersistence dist_pers;
    const auto start_distributed = std::chrono::high_resolution_clock::now();
    const auto distributed_pairs = dist_pers.compute(test_data);
    for (const auto &[birth, death, dim] : distributed_pairs)
    {
        const bool finite_or_essential_death =
            std::isfinite(death) || death == std::numeric_limits<float>::infinity();
        if (!std::isfinite(birth) || !finite_or_essential_death || dim < 0)
        {
            throw std::runtime_error("distributed benchmark produced invalid persistence output");
        }
    }
    const auto end_distributed = std::chrono::high_resolution_clock::now();
    bench.distributed_time_ms =
        std::chrono::duration<double, std::milli>(end_distributed - start_distributed).count();

    bench.single_node_time_ms = measureSingleNodeReductionMs(test_data);
    bench.speedup = finiteBenchmarkSpeedup(bench.single_node_time_ms, bench.distributed_time_ms);
    return bench;
}

} // namespace nerve::distributed
