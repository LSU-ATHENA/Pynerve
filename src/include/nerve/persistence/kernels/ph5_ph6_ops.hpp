
#pragma once

#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/errors/errors.hpp"
#include "nerve/persistence/utils/api.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nerve::persistence
{

template <typename T>
using Field = T;

namespace detail
{

template <typename Coord>
std::optional<double> coordToDouble(const Coord &value);

void appendU64(std::vector<uint8_t> &data, uint64_t value);

std::array<uint8_t, 32> hashBytes256(const std::vector<uint8_t> &data);

} // namespace detail

template <typename PointType, typename Scalar = double>
class PH5PH6Engine
{
public:
    using PointContainer = std::vector<PointType>;
    using ResultType = std::vector<std::pair<std::vector<size_t>, Scalar>>;

    struct Config
    {
        double numerical_tolerance = 1e-12;
        size_t max_iterations = 10000;
        bool validate_results = true;
        core::DeterminismLevel determinism_level = core::DeterminismLevel::BASIC;
        bool require_bitwise_reproducibility = false;
        bool enable_checksum_validation = true;
        std::string computation_id;
        std::array<uint64_t, 8> algorithm_seeds = {42, 43, 44, 45, 46, 47, 48, 49};
    };

    struct ComputationMetrics
    {
        double computation_time_ms = 0.0;
        size_t peak_memory_bytes = 0;
        size_t original_simplices = 0;
        size_t final_simplices = 0;
        double compression_ratio = 1.0;
        double quality_score = 1.0;
        bool passed_stability_checks = true;
        size_t numerical_errors = 0;
        std::array<uint8_t, 32> params_hash{};
        std::array<uint8_t, 32> result_checksum{};
        std::array<uint8_t, 16> rng_seed_used{};
        core::DeterminismLevel achieved_determinism_level = core::DeterminismLevel::BASIC;
        bool checksum_validation_passed = true;
        std::vector<std::string> determinism_warnings;
    };

    explicit PH5PH6Engine(const Config &config = Config{});

    std::optional<ResultType> computePersistenceCohomology(const PointContainer &points,
                                                           size_t max_dimension);
    std::optional<ResultType> computePersistenceCompressedWitness(const PointContainer &points,
                                                                  size_t max_dimension);
    std::optional<ResultType> computePersistenceBlockSparse(const PointContainer &points,
                                                            size_t max_dimension);
    std::optional<ResultType> computePersistenceHybrid(const PointContainer &points,
                                                       size_t max_dimension);

    ComputationMetrics getComputationMetrics() const;
    std::vector<std::pair<std::string, errors::ErrorCode>> getErrorLog() const;
    bool hasErrors() const;
    void clearErrorLog();
    std::string getLastError() const;

    bool validateDeterministicResult(const PointContainer &points, size_t max_dimension);
    bool runStabilityTest(const PointContainer &points, size_t max_dimension, size_t num_runs = 10);

    bool setDeterminismContract(const core::DeterminismContract &contract);
    core::DeterminismContract getDeterminismContract() const;
    bool validateDeterminismContract() const;

    std::array<uint8_t, 32> computeResultChecksum(const ResultType &result) const;
    bool validateResultChecksum(const ResultType &result,
                                const std::array<uint8_t, 32> &expected_checksum) const;
    void computeParamsHash(const PointContainer &points, size_t max_dimension);
    std::array<uint8_t, 32> getParamsHash() const;

private:
    struct FlattenedPoints
    {
        std::vector<double> values;
        Size point_dim = 0;
        Size num_points = 0;
    };

    static std::optional<FlattenedPoints> flattenPoints(const PointContainer &points,
                                                        std::string &error_message);

    using EntryPoint = errors::ErrorResult<PersistenceResult> (*)(
        const core::BufferView<const double> &, Size, const PersistenceOptions &);

    std::optional<ResultType> runCanonical(const PointContainer &points, size_t max_dimension,
                                           EntryPoint entrypoint);
    void recordError(const std::string &message, errors::ErrorCode code);

    Config config_;
    ComputationMetrics metrics_;
    core::DeterminismContract determinism_contract_;
    std::vector<std::pair<std::string, errors::ErrorCode>> error_log_;
    std::array<uint64_t, 8> algorithm_seeds_{};
    std::array<uint8_t, 32> current_params_hash_{};
};

template <typename PointType, typename Scalar = double>
std::unique_ptr<PH5PH6Engine<PointType, Scalar>> createPh5Engine();

template <typename PointType, typename Scalar = double>
std::unique_ptr<PH5PH6Engine<PointType, Scalar>> createPh6Engine();

template <typename PointType, typename Scalar = double>
std::unique_ptr<PH5PH6Engine<PointType, Scalar>> createPh5EngineIfEnabled();

template <typename PointType, typename Scalar = double>
std::unique_ptr<PH5PH6Engine<PointType, Scalar>> createPh6EngineIfEnabled();

template <typename PointType, typename Scalar = double>
bool isPh5Available();

template <typename PointType, typename Scalar = double>
bool isPh6Available();

} // namespace nerve::persistence

#include "nerve/persistence/kernels/ph5_ph6_methods_impl.hpp"
#include "nerve/persistence/kernels/ph5_ph6_runtime_impl.hpp"
