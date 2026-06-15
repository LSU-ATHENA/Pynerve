#pragma once
#include <memory>
#include <string>
#include <vector>

namespace nerve::encoders
{
enum class EncoderType
{
    CNN,
    MLP,
    TOPOLOGICAL,
    PERSISTENCE,
    GRAPH,
    HYBRID
};

class EncoderConfig
{
public:
    EncoderType type;
    std::vector<int> layers;
    double learning_rate = 0.001;
    size_t input_dim = 0;
    size_t output_dim = 0;
    bool validate() const;
};

class EncoderFactory
{
public:
    static std::unique_ptr<EncoderConfig> createConfig(EncoderType type);
    static bool isValidConfig(const EncoderConfig &config);
};

namespace utils
{
std::vector<double> normalize(const std::vector<double> &data, double min, double max);
std::vector<double> standardize(const std::vector<double> &data);
std::vector<double> applyNoise(const std::vector<double> &data, double noise_level);
} // namespace utils

class CNNEncoderConfig : public EncoderConfig
{
public:
    CNNEncoderConfig();
    void addConvLayer(int filters, int kernel_size);
};

class MLPEncoderConfig : public EncoderConfig
{
public:
    MLPEncoderConfig();
    void addLayer(int units);
};

class PersistenceEncoderConfig : public EncoderConfig
{
public:
    PersistenceEncoderConfig();
    void setStrategy(const std::string &strategy);
};
} // namespace nerve::encoders
