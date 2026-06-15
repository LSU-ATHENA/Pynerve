
#include "nerve/encoders/encoders.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nerve::encoders
{
namespace
{

std::string trim(std::string value)
{
    const auto first =
        std::ranges::find_if(value, [](unsigned char ch) { return !std::isspace(ch); });
    const auto last = std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
                          return !std::isspace(ch);
                      }).base();
    if (first >= last)
    {
        return {};
    }
    return std::string(first, last);
}

std::string canonicalType(std::string type)
{
    type = trim(std::move(type));
    std::ranges::transform(type, type.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (type == "cnn")
    {
        return "CNN";
    }
    if (type == "mlp")
    {
        return "MLP";
    }
    if (type == "topological" || type == "complex")
    {
        return "Topological";
    }
    if (type == "persistence")
    {
        return "Persistence";
    }
    if (type == "graph")
    {
        return "Graph";
    }
    if (type == "hybrid")
    {
        return "Hybrid";
    }
    return {};
}

std::map<std::string, std::string> parseConfigFile(const std::string &config_file)
{
    std::ifstream input(config_file);
    if (!input)
    {
        throw std::invalid_argument("Unable to open encoder config: " + config_file);
    }

    std::map<std::string, std::string> config;
    std::string line;
    Size line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos)
        {
            line.resize(comment);
        }
        line = trim(std::move(line));
        if (line.empty())
        {
            continue;
        }
        const auto delimiter = line.find('=');
        if (delimiter == std::string::npos)
        {
            throw std::invalid_argument("Invalid encoder config line " +
                                        std::to_string(line_number) + ": " + line);
        }
        std::string key = trim(line.substr(0, delimiter));
        std::string value = trim(line.substr(delimiter + 1));
        if (key.empty() || value.empty())
        {
            throw std::invalid_argument("Invalid encoder config line " +
                                        std::to_string(line_number) + ": " + line);
        }
        std::ranges::transform(
            key, key.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        config[key] = value;
    }
    return config;
}

std::string lookupString(const std::map<std::string, std::string> &config, const std::string &key,
                         const std::string &default_value = {})
{
    const auto it = config.find(key);
    return it == config.end() ? default_value : it->second;
}

Size lookupSize(const std::map<std::string, std::string> &config, const std::string &key,
                Size default_value)
{
    const auto it = config.find(key);
    if (it == config.end())
    {
        return default_value;
    }
    if (!it->second.empty() && it->second.front() == '-')
    {
        throw std::invalid_argument("Invalid unsigned integer for encoder config key: " + key);
    }
    std::size_t consumed = 0;
    const auto parsed = std::stoull(it->second, &consumed);
    if (consumed != it->second.size())
    {
        throw std::invalid_argument("Invalid integer for encoder config key: " + key);
    }
    if (parsed > std::numeric_limits<Size>::max())
    {
        throw std::out_of_range("Encoder config integer exceeds Size range: " + key);
    }
    return static_cast<Size>(parsed);
}

double lookupDouble(const std::map<std::string, std::string> &config, const std::string &key,
                    double default_value)
{
    const auto it = config.find(key);
    if (it == config.end())
    {
        return default_value;
    }
    std::size_t consumed = 0;
    const double parsed = std::stod(it->second, &consumed);
    if (consumed != it->second.size() || !std::isfinite(parsed))
    {
        throw std::invalid_argument("Invalid floating-point value for encoder config key: " + key);
    }
    return parsed;
}

std::vector<Size> lookupSizeList(const std::map<std::string, std::string> &config,
                                 const std::string &key, const std::vector<Size> &default_value)
{
    const auto it = config.find(key);
    if (it == config.end())
    {
        return default_value;
    }

    std::vector<Size> values;
    std::stringstream stream(it->second);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        token = trim(std::move(token));
        if (!token.empty())
        {
            if (token.front() == '-')
            {
                throw std::invalid_argument(
                    "Invalid unsigned integer list for encoder config key: " + key);
            }
            std::size_t consumed = 0;
            const auto parsed = std::stoull(token, &consumed);
            if (consumed != token.size())
            {
                throw std::invalid_argument("Invalid integer list for encoder config key: " + key);
            }
            values.push_back(static_cast<Size>(parsed));
        }
    }
    if (values.empty())
    {
        throw std::invalid_argument("Encoder config list must not be empty: " + key);
    }
    return values;
}

std::vector<std::string> lookupStringList(const std::map<std::string, std::string> &config,
                                          const std::string &key,
                                          const std::vector<std::string> &default_value)
{
    const auto it = config.find(key);
    if (it == config.end())
    {
        return default_value;
    }

    std::vector<std::string> values;
    std::stringstream stream(it->second);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        token = trim(std::move(token));
        if (!token.empty())
        {
            values.push_back(token);
        }
    }
    if (values.empty())
    {
        throw std::invalid_argument("Encoder config list must not be empty: " + key);
    }
    return values;
}

} // namespace

std::unique_ptr<FeatureEncoder>
EncoderFactory::createCnnEncoder(Size input_channels, Size output_channels,
                                 const std::vector<Size> &kernel_sizes)
{
    return std::make_unique<CNNEncoder>(input_channels, output_channels, kernel_sizes);
}
std::unique_ptr<FeatureEncoder> EncoderFactory::createMlpEncoder(Size input_size, Size hidden_size,
                                                                 Size output_size, Size num_layers)
{
    return std::make_unique<MLPEncoder>(input_size, hidden_size, output_size, num_layers);
}
std::unique_ptr<FeatureEncoder> EncoderFactory::createTopologicalEncoder(Size feature_dim)
{
    return std::make_unique<TopologicalEncoder>(feature_dim);
}
std::unique_ptr<FeatureEncoder> EncoderFactory::createPersistenceEncoder(Size output_dim)
{
    return std::make_unique<PersistenceEncoder>(output_dim);
}
std::unique_ptr<FeatureEncoder> EncoderFactory::createGraphEncoder(Size node_dim, Size edge_dim,
                                                                   Size output_dim)
{
    return std::make_unique<GraphEncoder>(node_dim, edge_dim, output_dim);
}
std::unique_ptr<FeatureEncoder>
EncoderFactory::createHybridEncoder(const std::vector<std::string> &encoder_types)
{
    return std::make_unique<HybridEncoder>(encoder_types);
}
std::unique_ptr<FeatureEncoder> EncoderFactory::createDefaultPersistenceEncoder()
{
    return createPersistenceEncoder(128);
}
std::unique_ptr<FeatureEncoder> EncoderFactory::createDefaultComplexEncoder()
{
    return createTopologicalEncoder(64);
}
std::unique_ptr<FeatureEncoder> EncoderFactory::createDefaultGraphEncoder()
{
    return createGraphEncoder(32, 16, 64);
}
std::unique_ptr<FeatureEncoder> EncoderFactory::createDefaultHybridEncoder()
{
    std::vector<std::string> encoder_types = {"CNN", "MLP", "Topological"};
    return createHybridEncoder(encoder_types);
}
std::unique_ptr<FeatureEncoder>
EncoderFactory::loadEncoderFromConfig(const std::string &config_file)
{
    const auto config = parseConfigFile(config_file);
    const std::string type =
        canonicalType(lookupString(config, "type", lookupString(config, "encoder_type")));
    if (type.empty())
    {
        throw std::invalid_argument("Encoder config requires a supported type");
    }

    if (type == "CNN")
    {
        return createCnnEncoder(
            lookupSize(config, "input_channels", lookupSize(config, "input_size", 1)),
            lookupSize(config, "output_channels", lookupSize(config, "output_size", 32)),
            lookupSizeList(config, "kernel_sizes", {3}));
    }
    if (type == "MLP")
    {
        return createMlpEncoder(
            lookupSize(config, "input_size", 1), lookupSize(config, "hidden_size", 128),
            lookupSize(config, "output_size", 64), lookupSize(config, "num_layers", 2));
    }
    if (type == "Topological")
    {
        auto encoder = createTopologicalEncoder(
            lookupSize(config, "feature_dim", lookupSize(config, "output_size", 64)));
        std::map<std::string, double> params;
        if (config.contains("landscape_resolution"))
        {
            params["landscape_resolution"] = lookupDouble(config, "landscape_resolution", 100.0);
        }
        if (config.contains("image_resolution"))
        {
            params["image_resolution"] = lookupDouble(config, "image_resolution", 50.0);
        }
        if (config.contains("image_sigma"))
        {
            params["image_sigma"] = lookupDouble(config, "image_sigma", 1.0);
        }
        if (!params.empty())
        {
            encoder->setParameters(params);
        }
        return encoder;
    }
    if (type == "Persistence")
    {
        return createPersistenceEncoder(
            lookupSize(config, "output_dim", lookupSize(config, "output_size", 128)));
    }
    if (type == "Graph")
    {
        return createGraphEncoder(
            lookupSize(config, "node_dim", 32), lookupSize(config, "edge_dim", 16),
            lookupSize(config, "output_dim", lookupSize(config, "output_size", 64)));
    }
    if (type == "Hybrid")
    {
        return createHybridEncoder(
            lookupStringList(config, "encoder_types", {"CNN", "MLP", "Topological"}));
    }
    throw std::invalid_argument("Unsupported encoder type: " + type);
}
void EncoderFactory::saveEncoderConfig(const FeatureEncoder &encoder,
                                       const std::string &config_file)
{
    std::ofstream output(config_file);
    if (!output)
    {
        throw std::invalid_argument("Unable to write encoder config: " + config_file);
    }

    const std::string type = encoder.getEncoderType();
    output << "type=" << type << '\n';
    output << "input_size=" << encoder.getInputSize() << '\n';
    output << "output_size=" << encoder.getOutputSize() << '\n';
    for (const auto &[key, value] : loadDefaultParams(type))
    {
        output << key << '=' << value << '\n';
    }
}
std::map<std::string, double> EncoderFactory::loadDefaultParams(const std::string &encoder_type)
{
    std::map<std::string, double> params;
    if (encoder_type == "CNN")
    {
        params["learning_rate"] = 0.001;
        params["dropout_rate"] = 0.5;
    }
    else if (encoder_type == "MLP")
    {
        params["learning_rate"] = 0.01;
        params["hidden_size"] = 128;
    }
    else if (encoder_type == "Topological")
    {
        params["landscape_resolution"] = 100;
        params["image_resolution"] = 50;
    }
    return params;
}
void EncoderFactory::validateEncoderConfig(const std::map<std::string, double> &params)
{
    for (const auto &[key, value] : params)
    {
        if (value < 0 || !std::isfinite(value))
        {
            throw std::invalid_argument("Parameter " + key + " must be finite and non-negative");
        }
    }
}

} // namespace nerve::encoders
