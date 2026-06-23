
#include "nerve/core/rng/determinism_contract.hpp"
#include "nerve/core/sha256.hpp"
#include "rng_wire_format.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

using nerve::core::SHA256_CTX;
using nerve::core::sha256Final;
using nerve::core::sha256Init;
using nerve::core::sha256Update;

namespace nerve::core
{

namespace
{

[[nodiscard]] uint32_t checkedLength(std::size_t size, const char *field)
{
    if (size > std::numeric_limits<uint32_t>::max())
    {
        throw std::length_error(field);
    }
    return static_cast<uint32_t>(size);
}

auto atOffset(const std::vector<uint8_t> &data, std::size_t offset)
{
    return std::next(data.begin(), static_cast<std::ptrdiff_t>(offset));
}

template <typename OutputIt>
void copyFromOffset(const std::vector<uint8_t> &data, std::size_t offset, std::size_t count,
                    OutputIt output)
{
    std::copy_n(atOffset(data, offset), static_cast<std::ptrdiff_t>(count), output);
}

} // namespace

DeterminismContract::DeterminismContract()
    : level(DeterminismLevel::BASIC)
    , start_time(std::chrono::steady_clock::now())
{
    enable_checksum_validation = true;
    enable_deterministic_threading = false;
    enable_deterministic_random = true;
    rng_seed_provided = true;
    rng_seed.fill(0);
    component_name = "default";
}

DeterminismContract::DeterminismContract(DeterminismLevel level, const std::string &component)
    : level(level)
    , component_name(component)
    , start_time(std::chrono::steady_clock::now())
{
    switch (level)
    {
        case DeterminismLevel::AUDIT:
            enable_audit_trail = true;
            record_intermediate_results = true;
            validate_numerical_stability = true;
            enable_checksum_validation = true;
            fail_on_non_deterministic = true;
            break;
        case DeterminismLevel::STRICT:
            enable_deterministic_threading = true;
            enable_deterministic_random = true;
            [[fallthrough]];
        case DeterminismLevel::BASIC:
        case DeterminismLevel::NONE:
            enable_checksum_validation = true;
            break;
    }
}
[[nodiscard]] bool DeterminismContract::isValid() const
{
    return validationErrors().empty();
}
std::vector<std::string> DeterminismContract::validationErrors() const
{
    std::vector<std::string> errors;
    if (level > DeterminismLevel::AUDIT)
    {
        errors.push_back("Invalid determinism level");
    }
    if (component_name.empty() && level >= DeterminismLevel::BASIC)
    {
        errors.push_back("Component name required for determinism level BASIC and above");
    }
    if (max_execution_time.count() <= 0)
    {
        errors.push_back("Max execution time must be positive");
    }
    if (max_memory_usage_mb == 0)
    {
        errors.push_back("Max memory usage must be positive");
    }
    if (enable_deterministic_random && !rng_seed_provided)
    {
        errors.push_back("RNG seed required when deterministic random is enabled");
    }
    if (enable_checksum_validation && !params_hash_valid && level >= DeterminismLevel::STRICT)
    {
        errors.push_back("Params hash required when checksum validation is enabled");
    }
    if (enable_deterministic_threading && !DeterminismEnforcer::supportsDeterministicThreading())
    {
        errors.push_back("Deterministic threading not supported on this system");
    }
    return errors;
}
void DeterminismContract::setRngSeed(uint64_t seed)
{
    rng_seed.fill(0);
    for (size_t i = 0; i < 8 && i < rng_seed.size(); ++i)
    {
        rng_seed[i] = static_cast<uint8_t>(seed >> (i * 8));
    }
    rng_seed_provided = true;
}
void DeterminismContract::setRngSeed(const std::array<uint8_t, 16> &seed)
{
    rng_seed = seed;
    rng_seed_provided = true;
}
void DeterminismContract::computeParamsHash(
    const std::vector<std::pair<std::string, std::string>> &params)
{
    SHA256_CTX ctx;
    sha256Init(&ctx);
    std::vector<std::pair<std::string, std::string>> sorted_params = params;
    std::ranges::sort(sorted_params);
    for (const auto &[key, value] : sorted_params)
    {
        sha256Update(&ctx, reinterpret_cast<const unsigned char *>(key.c_str()), key.length());
        sha256Update(&ctx, reinterpret_cast<const unsigned char *>("\0"), 1);
        sha256Update(&ctx, reinterpret_cast<const unsigned char *>(value.c_str()), value.length());
        sha256Update(&ctx, reinterpret_cast<const unsigned char *>("\0"), 1);
    }
    sha256Final(&ctx, params_hash.data());
    params_hash_valid = true;
}
std::vector<uint8_t> DeterminismContract::serialize() const
{
    std::vector<uint8_t> data;
    data.push_back(static_cast<uint8_t>(level));
    uint8_t flags = 0;
    flags |= static_cast<uint8_t>(enable_checksum_validation) << 0;
    flags |= static_cast<uint8_t>(enable_deterministic_threading) << 1;
    flags |= static_cast<uint8_t>(enable_deterministic_random) << 2;
    flags |= static_cast<uint8_t>(rng_seed_provided) << 3;
    flags |= static_cast<uint8_t>(params_hash_valid) << 4;
    flags |= static_cast<uint8_t>(enable_audit_trail) << 5;
    flags |= static_cast<uint8_t>(record_intermediate_results) << 6;
    flags |= static_cast<uint8_t>(validate_numerical_stability) << 7;
    data.push_back(flags);
    data.insert(data.end(), rng_seed.begin(), rng_seed.end());
    data.insert(data.end(), params_hash.begin(), params_hash.end());
    uint64_t time_ms = max_execution_time.count();
    detail::appendUint64LittleEndian(data, time_ms);
    uint64_t memory_mb = max_memory_usage_mb;
    detail::appendUint64LittleEndian(data, memory_mb);
    uint32_t name_len =
        checkedLength(component_name.length(), "Determinism component name too long");
    detail::appendUint32LittleEndian(data, name_len);
    data.insert(data.end(), component_name.begin(), component_name.end());
    return data;
}
bool DeterminismContract::deserialize(const std::vector<uint8_t> &data)
{
    if (data.empty())
        return false;
    size_t offset = 0;
    if (offset >= data.size())
        return false;
    level = static_cast<DeterminismLevel>(data[offset++]);
    if (offset >= data.size())
        return false;
    uint8_t flags = data[offset++];
    enable_checksum_validation = (flags & (1 << 0)) != 0;
    enable_deterministic_threading = (flags & (1 << 1)) != 0;
    enable_deterministic_random = (flags & (1 << 2)) != 0;
    rng_seed_provided = (flags & (1 << 3)) != 0;
    params_hash_valid = (flags & (1 << 4)) != 0;
    enable_audit_trail = (flags & (1 << 5)) != 0;
    record_intermediate_results = (flags & (1 << 6)) != 0;
    validate_numerical_stability = (flags & (1 << 7)) != 0;
    if (offset + rng_seed.size() > data.size())
        return false;
    copyFromOffset(data, offset, rng_seed.size(), rng_seed.begin());
    offset += rng_seed.size();
    if (offset + params_hash.size() > data.size())
        return false;
    copyFromOffset(data, offset, params_hash.size(), params_hash.begin());
    offset += params_hash.size();
    uint64_t time_ms = 0;
    if (!detail::readUint64LittleEndian(data, &offset, &time_ms))
        return false;
    max_execution_time = std::chrono::milliseconds(time_ms);
    uint64_t memory_mb = 0;
    if (!detail::readUint64LittleEndian(data, &offset, &memory_mb))
        return false;
    max_memory_usage_mb = memory_mb;
    uint32_t name_len = 0;
    if (!detail::readUint32LittleEndian(data, &offset, &name_len))
        return false;
    if (offset + name_len > data.size())
        return false;
    component_name.assign(reinterpret_cast<const char *>(data.data() + offset), name_len);
    return true;
}
bool DeterminismEnforcer::canSatisfyContract(const DeterminismContract &contract)
{
    return getContractViolations(contract).empty();
}
std::vector<std::string>
DeterminismEnforcer::getContractViolations(const DeterminismContract &contract)
{
    std::vector<std::string> violations;
    if (contract.enable_deterministic_threading && !supportsDeterministicThreading())
    {
        violations.push_back("Deterministic threading required but not supported");
    }
    if (contract.level >= DeterminismLevel::STRICT && !supportsBitwiseReproducibility())
    {
        violations.push_back("Bitwise reproducibility required but not supported");
    }
    if (contract.enable_deterministic_random && !contract.rng_seed_provided)
    {
        violations.push_back("Deterministic random required but no seed provided");
    }
    if (contract.level >= DeterminismLevel::STRICT)
    {
        violations.push_back("Strict determinism requires careful floating-point handling");
    }
    return violations;
}
DeterminismContract DeterminismEnforcer::createContract(DeterminismLevel level,
                                                        const std::string &component_name)
{
    return DeterminismContract(level, component_name);
}
bool DeterminismEnforcer::supportsDeterministicThreading()
{
    return std::thread::hardware_concurrency() <= 1;
}
bool DeterminismEnforcer::supportsDeterministicGpu()
{
    return false;
}
bool DeterminismEnforcer::supportsBitwiseReproducibility()
{
    return supportsDeterministicThreading() && !supportsDeterministicGpu();
}
DeterminismEnforcer::SystemCapabilities DeterminismEnforcer::getSystemCapabilities()
{
    SystemCapabilities caps;
    caps.deterministic_threading = supportsDeterministicThreading();
    caps.deterministic_gpu = supportsDeterministicGpu();
    caps.bitwise_reproducibility = supportsBitwiseReproducibility();
    caps.hardware_random = false;
    caps.cpu_info = "x86_64";
    caps.gpu_info = caps.deterministic_gpu ? "CUDA" : "None";
    return caps;
}
bool DeterminismEnforcer::validateComputationResult(
    const DeterminismContract &contract, const std::vector<uint8_t> &result_data,
    const std::array<uint8_t, 32> &expected_checksum)
{
    if (!contract.enable_checksum_validation)
    {
        return true;
    }
    auto actual_checksum = computeChecksum(result_data);
    return actual_checksum == expected_checksum;
}
uint32_t DeterminismEnforcer::deterministicRandomUint32(const DeterminismContract &contract)
{
    if (!contract.enable_deterministic_random || !contract.rng_seed_provided)
    {
        return 0;
    }
    std::mt19937 generator;
    std::seed_seq seq(contract.rng_seed.begin(), contract.rng_seed.end());
    generator.seed(seq);
    return generator();
}
double DeterminismEnforcer::deterministicRandomDouble(const DeterminismContract &contract)
{
    if (!contract.enable_deterministic_random || !contract.rng_seed_provided)
    {
        return 0.0;
    }
    std::mt19937 generator;
    std::seed_seq seq(contract.rng_seed.begin(), contract.rng_seed.end());
    generator.seed(seq);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(generator);
}
std::array<uint8_t, 32> DeterminismEnforcer::computeChecksum(const std::vector<uint8_t> &data)
{
    std::array<uint8_t, 32> checksum;
    SHA256_CTX ctx;
    sha256Init(&ctx);
    sha256Update(&ctx, data.data(), data.size());
    sha256Final(&ctx, checksum.data());
    return checksum;
}
bool DeterminismEnforcer::validateRngSeed(const std::array<uint8_t, 16> &seed)
{
    for (uint8_t byte : seed)
    {
        if (byte != 0)
            return true;
    }
    return false;
}
DeterminismContext::DeterminismContext(const DeterminismContract &contract)
    : contract_(contract)
{
    activateContract();
}
DeterminismContext::~DeterminismContext()
{
    if (active_)
    {
        deactivateContract();
    }
}
DeterminismContext::DeterminismContext(DeterminismContext &&other) noexcept
    : contract_(std::move(other.contract_))
    , previous_contract_(std::move(other.previous_contract_))
    , active_(other.active_)
    , fail_on_non_deterministic_(other.fail_on_non_deterministic_)
{
    other.active_ = false;
}
DeterminismContext &DeterminismContext::operator=(DeterminismContext &&other) noexcept
{
    if (this != &other)
    {
        if (active_)
        {
            deactivateContract();
        }
        contract_ = std::move(other.contract_);
        previous_contract_ = std::move(other.previous_contract_);
        active_ = other.active_;
        fail_on_non_deterministic_ = other.fail_on_non_deterministic_;
        other.active_ = false;
    }
    return *this;
}
const DeterminismContract &DeterminismContext::contract() const
{
    return contract_;
}
bool DeterminismContext::isActive() const
{
    return active_;
}
void DeterminismContext::setFailOnNonDeterministic(bool fail)
{
    fail_on_non_deterministic_ = fail;
}
void DeterminismContext::activateContract()
{
    previous_contract_ = contract_;
    if (!contract_.isValid())
    {
        auto errors = contract_.validationErrors();
        if (fail_on_non_deterministic_)
        {
            throw std::runtime_error("Invalid determinism contract: " + errors[0]);
        }
        return;
    }
    auto violations = DeterminismEnforcer::getContractViolations(contract_);
    if (!violations.empty() && fail_on_non_deterministic_)
    {
        throw std::runtime_error("Cannot satisfy determinism contract: " + violations[0]);
    }
    active_ = true;
}
void DeterminismContext::deactivateContract()
{
    active_ = false;
}
void DeterminismContract::finalizeParamsHash()
{
    params_hash_valid = true;
}
template <typename T>
void DeterminismContract::addParameterToHash(const std::string &name, const T &value)
{
    std::ostringstream oss;
    if constexpr (std::is_arithmetic_v<T>)
    {
        oss << name << "=" << static_cast<double>(value);
    }
    else
    {
        oss << name << "=" << value;
    }
    computeParamsHash({{name, oss.str()}});
}
DeterminismContract DeterminismEnforcer::create(uint64_t seed, uint64_t params_hash,
                                                bool bitwise_reproducible, uint8_t precision_level,
                                                uint8_t algorithm_version)
{
    DeterminismContract contract;
    contract.level = bitwise_reproducible ? DeterminismLevel::STRICT : DeterminismLevel::BASIC;
    contract.enable_checksum_validation = true;
    contract.enable_deterministic_random = true;
    contract.fail_on_non_deterministic = bitwise_reproducible;
    for (std::size_t i = 0; i < std::min(sizeof(seed), contract.rng_seed.size()); ++i)
    {
        contract.rng_seed[i] = static_cast<uint8_t>(seed >> (i * 8));
    }
    contract.rng_seed_provided = true;
    for (std::size_t i = 0; i < std::min(sizeof(params_hash), contract.params_hash.size()); ++i)
    {
        contract.params_hash[i] = static_cast<uint8_t>(params_hash >> (i * 8));
    }
    if (contract.params_hash.size() > sizeof(params_hash) + 1)
    {
        contract.params_hash[sizeof(params_hash)] = precision_level;
        contract.params_hash[sizeof(params_hash) + 1] = algorithm_version;
    }
    contract.params_hash_valid = true;
    return contract;
}
} // namespace nerve::core
