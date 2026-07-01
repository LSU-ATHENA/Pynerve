
#include "nerve/encoders/detail/encoders_detail.hpp"
#include "nerve/encoders/encoders.hpp"

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace
{

using nerve::Size;

bool check_encoder_factory_creates_cnn()
{
    auto enc = nerve::encoders::EncoderFactory::createCnnEncoder(3, 16, {3, 3});
    if (!enc)
    {
        return false;
    }
    return enc->getEncoderType() == "CNN";
}

bool check_encoder_factory_creates_mlp()
{
    auto enc = nerve::encoders::EncoderFactory::createMlpEncoder(64, 128, 10, 2);
    if (!enc)
    {
        return false;
    }
    return enc->getEncoderType() == "MLP";
}

bool check_encoder_factory_creates_topological()
{
    auto enc = nerve::encoders::EncoderFactory::createTopologicalEncoder(32);
    if (!enc)
    {
        return false;
    }
    return enc->getEncoderType() == "Topological";
}

bool check_encoder_factory_creates_persistence()
{
    auto enc = nerve::encoders::EncoderFactory::createPersistenceEncoder(128);
    if (!enc)
    {
        return false;
    }
    return enc->getEncoderType() == "Persistence";
}

bool check_encoder_factory_creates_graph()
{
    auto enc = nerve::encoders::EncoderFactory::createGraphEncoder(32, 16, 64);
    if (!enc)
    {
        return false;
    }
    return enc->getEncoderType() == "Graph";
}

bool check_encoder_factory_creates_hybrid()
{
    std::vector<std::string> types = {"CNN", "MLP", "Topological"};
    auto enc = nerve::encoders::EncoderFactory::createHybridEncoder(types);
    if (!enc)
    {
        return false;
    }
    return enc->getEncoderType() == "Hybrid";
}

bool check_encoder_factory_defaults()
{
    auto pe = nerve::encoders::EncoderFactory::createDefaultPersistenceEncoder();
    if (!pe || pe->getEncoderType() != "Persistence")
    {
        return false;
    }
    auto ge = nerve::encoders::EncoderFactory::createDefaultGraphEncoder();
    if (!ge || ge->getEncoderType() != "Graph")
    {
        return false;
    }
    auto ce = nerve::encoders::EncoderFactory::createDefaultComplexEncoder();
    if (!ce || ce->getEncoderType() != "Topological")
    {
        return false;
    }
    return true;
}

bool check_encoder_factory_default_params()
{
    auto params = nerve::encoders::EncoderFactory::loadDefaultParams("CNN");
    auto it = params.find("learning_rate");
    if (it == params.end())
    {
        return false;
    }
    return std::abs(it->second - 0.001) < 1e-9;
}

bool check_encoder_factory_validate_params()
{
    std::map<std::string, double> valid = {{"dropout", 0.5}, {"lr", 0.01}};
    try
    {
        nerve::encoders::EncoderFactory::validateEncoderConfig(valid);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool check_encoder_factory_invalid_params()
{
    std::map<std::string, double> invalid = {{"rate", -1.0}};
    try
    {
        nerve::encoders::EncoderFactory::validateEncoderConfig(invalid);
        return false;
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
}

bool check_encoder_utils_normalize()
{
    std::vector<std::vector<double>> data = {{0.0, 10.0}, {5.0, 20.0}};
    auto norm = nerve::encoders::EncoderUtils::normalizeData(data);
    if (norm.size() != 2)
    {
        return false;
    }
    if (norm[0].size() != 2)
    {
        return false;
    }
    if (std::abs(norm[0][0] - 0.0) > 1e-12 || std::abs(norm[1][0] - 1.0) > 1e-12)
    {
        return false;
    }
    return true;
}

bool check_encoder_utils_standardize()
{
    std::vector<std::vector<double>> data = {{1.0, 2.0}, {3.0, 4.0}};
    auto std_data = nerve::encoders::EncoderUtils::standardizeData(data);
    if (std_data.size() != 2)
    {
        return false;
    }
    return std_data[0].size() == 2;
}

bool check_cnn_encoder_config()
{
    nerve::encoders::CNNEncoder cnn(3, 16, {3});
    Size input = cnn.getInputSize();
    Size output = cnn.getOutputSize();
    std::string type = cnn.getEncoderType();

    if (input != 3)
    {
        return false;
    }
    if (output != 16)
    {
        return false;
    }
    if (type != "CNN")
    {
        return false;
    }
    return true;
}

bool check_cnn_encoder_add_layers()
{
    nerve::encoders::CNNEncoder cnn(1, 8, {3});
    cnn.addConvLayer(8, 16, 3);
    cnn.addPoolingLayer(2, "max");
    cnn.addActivationLayer("relu");
    (void)cnn.getInputSize();
    (void)cnn.getOutputSize();
    return true;
}

bool check_mlp_encoder_config()
{
    nerve::encoders::MLPEncoder mlp(64, 128, 10, 2);
    Size input = mlp.getInputSize();
    Size output = mlp.getOutputSize();
    std::string type = mlp.getEncoderType();

    if (input != 64)
    {
        return false;
    }
    if (output != 10)
    {
        return false;
    }
    if (type != "MLP")
    {
        return false;
    }
    return true;
}

bool check_mlp_encoder_add_layers()
{
    nerve::encoders::MLPEncoder mlp(16, 32, 8, 2);
    mlp.addLayer(32, 64, "relu");
    mlp.addBatchNormLayer();
    mlp.addDropoutLayer(0.5);
    (void)mlp.getInputSize();
    (void)mlp.getOutputSize();
    return true;
}

bool check_persistence_encoder_basic_config()
{
    nerve::encoders::PersistenceEncoder pe(64);
    Size output = pe.getOutputSize();
    std::string type = pe.getEncoderType();

    if (output != 64)
    {
        return false;
    }
    if (type != "Persistence")
    {
        return false;
    }

    pe.setEncodingStrategy("landscapes");
    pe.setLandscapesParams(3, 100);
    pe.setImagesParams(32, 1.5);
    pe.setStatisticsParams(true, true);
    return true;
}

bool check_encoder_utils_apply_noise()
{
    std::vector<std::vector<double>> data = {{1.0, 2.0}, {3.0, 4.0}};
    auto noisy = nerve::encoders::EncoderUtils::applyNoise(data, 0.1);
    if (noisy.size() != data.size())
    {
        return false;
    }
    if (noisy[0].size() != data[0].size())
    {
        return false;
    }
    return true;
}

bool check_encoder_utils_output_dims()
{
    nerve::encoders::CNNEncoder cnn(1, 8, {3});
    nerve::encoders::MLPEncoder mlp(10, 32, 5, 2);
    nerve::encoders::PersistenceEncoder pe(16);

    if (cnn.getOutputSize() != 8)
    {
        return false;
    }
    if (mlp.getOutputSize() != 5)
    {
        return false;
    }
    if (pe.getOutputSize() != 16)
    {
        return false;
    }
    return true;
}

bool check_cnn_encoder_invalid_throws()
{
    try
    {
        nerve::encoders::CNNEncoder bad(0, 16, {3});
        return false;
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
}

bool check_mlp_encoder_invalid_throws()
{
    try
    {
        nerve::encoders::MLPEncoder bad(64, 0, 10, 2);
        return false;
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
}

} // namespace

int main()
{
    int failures = 0;

    auto run = [&](const char *name, bool ok) {
        if (!ok)
        {
            std::cerr << "FAIL: " << name << "\n";
            ++failures;
        }
        else
        {
            std::cout << "PASS: " << name << "\n";
        }
    };

    run("encoder_factory_creates_cnn", check_encoder_factory_creates_cnn());
    run("encoder_factory_creates_mlp", check_encoder_factory_creates_mlp());
    run("encoder_factory_creates_topological", check_encoder_factory_creates_topological());
    run("encoder_factory_creates_persistence", check_encoder_factory_creates_persistence());
    run("encoder_factory_creates_graph", check_encoder_factory_creates_graph());
    run("encoder_factory_creates_hybrid", check_encoder_factory_creates_hybrid());
    run("encoder_factory_defaults", check_encoder_factory_defaults());
    run("encoder_factory_default_params", check_encoder_factory_default_params());
    run("encoder_factory_validate_params", check_encoder_factory_validate_params());
    run("encoder_factory_invalid_params", check_encoder_factory_invalid_params());
    run("encoder_utils_normalize", check_encoder_utils_normalize());
    run("encoder_utils_standardize", check_encoder_utils_standardize());
    run("cnn_encoder_config", check_cnn_encoder_config());
    run("cnn_encoder_add_layers", check_cnn_encoder_add_layers());
    run("mlp_encoder_config", check_mlp_encoder_config());
    run("mlp_encoder_add_layers", check_mlp_encoder_add_layers());
    run("persistence_encoder_basic_config", check_persistence_encoder_basic_config());
    run("encoder_utils_apply_noise", check_encoder_utils_apply_noise());
    run("encoder_utils_output_dims", check_encoder_utils_output_dims());
    run("cnn_encoder_invalid_throws", check_cnn_encoder_invalid_throws());
    run("mlp_encoder_invalid_throws", check_mlp_encoder_invalid_throws());

    return failures > 0 ? 1 : 0;
}
