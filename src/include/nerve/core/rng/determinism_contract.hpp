
#pragma once
#include "nerve/config.hpp"

#include <array>
#include <chrono>
#include <compare>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace nerve::core
{

enum class DeterminismLevel : uint8_t
{
    NONE = 0,
    BASIC = 1,
    STRICT = 2,
    AUDIT = 3
};

struct DeterminismContract
{
    DeterminismLevel level = DeterminismLevel::BASIC;
    bool enable_checksum_validation = true;
    bool enable_deterministic_threading = false;
    bool enable_deterministic_random = true;
    std::array<uint8_t, 16> rng_seed = {};
    bool rng_seed_provided = false;
    std::array<uint8_t, 32> params_hash = {};
    bool params_hash_valid = false;
    std::chrono::milliseconds max_execution_time{1000};
    size_t max_memory_usage_mb = 1024;
    bool enable_audit_trail = false;
    bool record_intermediate_results = false;
    bool validate_numerical_stability = false;
    bool fail_on_non_deterministic = false;
    bool fail_on_contract_failure = true;
    std::string computation_id;
    std::string component_name;
    std::chrono::steady_clock::time_point start_time;

    [[nodiscard]] bool isValid() const;
    [[nodiscard]] std::vector<std::string> validationErrors() const;
    void setRngSeed(uint64_t seed);
    void setRngSeed(const std::array<uint8_t, 16> &seed);
    void computeParamsHash(const std::vector<std::pair<std::string, std::string>> &params);
    template <typename T>
    void addParameterToHash(const std::string &name, const T &value);
    void finalizeParamsHash();
    [[nodiscard]] std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t> &data);

    bool operator==(const DeterminismContract &other) const
    {
        return computation_id == other.computation_id && component_name == other.component_name &&
               start_time == other.start_time;
    }
    bool operator<(const DeterminismContract &other) const
    {
        return std::tie(computation_id, component_name, start_time) <
               std::tie(other.computation_id, other.component_name, other.start_time);
    }

    DeterminismContract();
    DeterminismContract(DeterminismLevel level, const std::string &component = "");
};

class DeterminismEnforcer
{
public:
    static bool canSatisfyContract(const DeterminismContract &contract);
    static std::vector<std::string> getContractViolations(const DeterminismContract &contract);
    static DeterminismContract createContract(DeterminismLevel level,
                                              const std::string &component_name = "");
    static DeterminismContract create(uint64_t seed, uint64_t params_hash,
                                      bool bitwise_reproducible = true,
                                      uint8_t precision_level = 255, uint8_t algorithm_version = 1);
    static bool supportsDeterministicThreading();
    static bool supportsDeterministicGpu();
    static bool supportsBitwiseReproducibility();
    struct SystemCapabilities
    {
        bool deterministic_threading = false;
        bool deterministic_gpu = false;
        bool bitwise_reproducibility = false;
        bool hardware_random = false;
        std::string cpu_info;
        std::string gpu_info;
    };
    static SystemCapabilities getSystemCapabilities();
    static bool validateComputationResult(const DeterminismContract &contract,
                                          const std::vector<uint8_t> &result_data,
                                          const std::array<uint8_t, 32> &expected_checksum);
    static uint32_t deterministicRandomUint32(const DeterminismContract &contract);
    static double deterministicRandomDouble(const DeterminismContract &contract);

private:
    static std::array<uint8_t, 32> computeChecksum(const std::vector<uint8_t> &data);
    static bool validateRngSeed(const std::array<uint8_t, 16> &seed);
};
class DeterminismContext
{
public:
    explicit DeterminismContext(const DeterminismContract &contract);
    ~DeterminismContext();
    DeterminismContext(const DeterminismContext &) = delete;
    DeterminismContext &operator=(const DeterminismContext &) = delete;
    DeterminismContext(DeterminismContext &&other) noexcept;
    DeterminismContext &operator=(DeterminismContext &&other) noexcept;
    const DeterminismContract &contract() const;
    bool isActive() const;
    void setFailOnNonDeterministic(bool fail);

private:
    DeterminismContract contract_;
    DeterminismContract previous_contract_;
    bool active_ = false;
    bool fail_on_non_deterministic_ = false;
    void activateContract();
    void deactivateContract();
};
struct DeterminismMetadata
{
    std::array<uint8_t, 32> params_hash;
    std::array<uint8_t, 32> result_checksum;
    std::array<uint8_t, 16> rng_seed_used;
    DeterminismLevel achieved_level;
    std::chrono::milliseconds actual_execution_time{0};
    size_t actual_memory_usage_mb = 0;
    bool was_deterministic = false;
    std::vector<std::string> warnings;
    std::string error_message;
    bool isValid() const;
    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t> &data);
};
} // namespace nerve::core
