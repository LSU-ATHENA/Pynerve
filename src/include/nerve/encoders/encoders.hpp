
#pragma once
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/autodiff/autodiff.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"

#include <functional>
#include <map>
#include <memory>
#include <vector>
namespace nerve::encoders
{
using algebra::Simplex;
using algebra::SimplicialComplex;
using autodiff::Tensor;
using autodiff::Variable;
using persistence::Diagram;
class FeatureEncoder;
class CNNEncoder;
class MLPEncoder;
class TopologicalEncoder;
class PersistenceEncoder;
class GraphEncoder;
class HybridEncoder;
class EncoderFactory;
class FeatureEncoder
{
public:
    FeatureEncoder() = default;
    virtual ~FeatureEncoder() = default;
    virtual Tensor encode(const std::vector<std::vector<double>> &data) const = 0;
    virtual Tensor encode(const SimplicialComplex &complex) const = 0;
    virtual Tensor encode(const Diagram &diagram) const = 0;
    virtual std::vector<Tensor>
    encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const = 0;
    virtual std::vector<Tensor>
    encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const = 0;
    virtual std::vector<Tensor> encodeBatch(const std::vector<Diagram> &batch_diagrams) const = 0;
    virtual void setInputSize(Size input_size) = 0;
    virtual void setOutputSize(Size output_size) = 0;
    virtual void setParameters(const std::map<std::string, double> &params) = 0;
    virtual Size getInputSize() const = 0;
    virtual Size getOutputSize() const = 0;
    virtual std::string getEncoderType() const = 0;

protected:
    Size input_size_;
    Size output_size_;
    std::map<std::string, double> parameters_;
};
class CNNEncoder : public FeatureEncoder
{
public:
    CNNEncoder(Size input_channels, Size output_channels, const std::vector<Size> &kernel_sizes);
    Tensor encode(const std::vector<std::vector<double>> &data) const override;
    Tensor encode(const SimplicialComplex &complex) const override;
    Tensor encode(const Diagram &diagram) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const override;
    std::vector<Tensor> encodeBatch(const std::vector<Diagram> &batch_diagrams) const override;
    void setInputSize(Size input_size) override;
    void setOutputSize(Size output_size) override;
    void setParameters(const std::map<std::string, double> &params) override;
    void addConvLayer(Size in_channels, Size out_channels, Size kernel_size);
    void addPoolingLayer(Size pool_size, const std::string &pool_type = "max");
    void addActivationLayer(const std::string &activation = "relu");
    void addDropoutLayer(double dropout_rate);
    Tensor forward(const Tensor &input) const;
    Size getInputSize() const override;
    Size getOutputSize() const override;
    std::string getEncoderType() const override;

private:
    struct ConvLayer
    {
        Size in_channels;
        Size out_channels;
        Size kernel_size;
        Tensor weights;
        Tensor bias;
    };
    struct PoolingLayer
    {
        Size pool_size;
        std::string pool_type;
    };
    struct ActivationLayer
    {
        std::string activation_type;
    };
    struct DropoutLayer
    {
        double dropout_rate;
    };
    std::vector<ConvLayer> conv_layers_;
    std::vector<PoolingLayer> pooling_layers_;
    std::vector<ActivationLayer> activation_layers_;
    std::vector<DropoutLayer> dropout_layers_;
    Tensor applyConvolution(const Tensor &input, const ConvLayer &layer) const;
    Tensor applyPooling(const Tensor &input, const PoolingLayer &layer) const;
    Tensor applyActivation(const Tensor &input, const ActivationLayer &layer) const;
    Tensor applyDropout(const Tensor &input, const DropoutLayer &layer) const;
    Tensor preprocessData(const std::vector<std::vector<double>> &data) const;
    Tensor preprocessComplex(const SimplicialComplex &complex) const;
    Tensor preprocessDiagram(const Diagram &diagram) const;
};
class MLPEncoder : public FeatureEncoder
{
public:
    MLPEncoder(Size input_size, Size hidden_size, Size output_size, Size num_layers);
    Tensor encode(const std::vector<std::vector<double>> &data) const override;
    Tensor encode(const SimplicialComplex &complex) const override;
    Tensor encode(const Diagram &diagram) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const override;
    std::vector<Tensor> encodeBatch(const std::vector<Diagram> &batch_diagrams) const override;
    void setInputSize(Size input_size) override;
    void setOutputSize(Size output_size) override;
    void setParameters(const std::map<std::string, double> &params) override;
    void addLayer(Size input_dim, Size output_dim, const std::string &activation = "relu");
    void addBatchNormLayer();
    void addDropoutLayer(double dropout_rate);
    Tensor forward(const Tensor &input) const;
    Size getInputSize() const override;
    Size getOutputSize() const override;
    std::string getEncoderType() const override;

private:
    struct LinearLayer
    {
        Size input_dim;
        Size output_dim;
        Tensor weights;
        Tensor bias;
        std::string activation;
    };
    struct BatchNormLayer
    {
        Tensor gamma;
        Tensor beta;
        Tensor running_mean;
        Tensor running_var;
        double momentum;
        double epsilon;
    };
    struct DropoutLayer
    {
        double dropout_rate;
    };
    std::vector<LinearLayer> linear_layers_;
    std::vector<BatchNormLayer> batch_norm_layers_;
    std::vector<DropoutLayer> dropout_layers_;
    Tensor applyLinear(const Tensor &input, const LinearLayer &layer) const;
    Tensor applyBatchNorm(const Tensor &input, const BatchNormLayer &layer) const;
    Tensor applyDropout(const Tensor &input, const DropoutLayer &layer) const;
    Tensor applyActivation(const Tensor &input, const std::string &activation) const;
    Tensor preprocessData(const std::vector<std::vector<double>> &data) const;
    Tensor preprocessComplex(const SimplicialComplex &complex) const;
    Tensor preprocessDiagram(const Diagram &diagram) const;
};
class TopologicalEncoder : public FeatureEncoder
{
public:
    TopologicalEncoder(Size feature_dim);
    Tensor encode(const std::vector<std::vector<double>> &data) const override;
    Tensor encode(const SimplicialComplex &complex) const override;
    Tensor encode(const Diagram &diagram) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const override;
    std::vector<Tensor> encodeBatch(const std::vector<Diagram> &batch_diagrams) const override;
    void setInputSize(Size input_size) override;
    void setOutputSize(Size output_size) override;
    void setParameters(const std::map<std::string, double> &params) override;
    void enableBettiNumbers(bool enable);
    void enablePersistenceLandscape(bool enable);
    void enablePersistenceImages(bool enable);
    void enablePersistenceEntropy(bool enable);
    std::vector<int> extractBettiNumbers(const SimplicialComplex &complex) const;
    std::vector<double> extractPersistenceLandscape(const Diagram &diagram) const;
    std::vector<double> extractPersistenceImages(const Diagram &diagram) const;
    std::vector<double> extractPersistenceEntropy(const Diagram &diagram) const;
    Size getInputSize() const override;
    Size getOutputSize() const override;
    std::string getEncoderType() const override;

private:
    bool enable_betti_numbers_;
    bool enable_persistence_landscape_;
    bool enable_persistence_images_;
    bool enable_persistence_entropy_;
    Size landscape_resolution_;
    Size image_resolution_;
    double image_sigma_;
    std::vector<double> computePersistenceLandscape(const Diagram &diagram, Size resolution) const;
    std::vector<double> computePersistenceImage(const Diagram &diagram, Size resolution,
                                                double sigma) const;
    double computePersistenceEntropy(const Diagram &diagram) const;
};
class PersistenceEncoder : public FeatureEncoder
{
public:
    PersistenceEncoder(Size output_dim);
    Tensor encode(const std::vector<std::vector<double>> &data) const override;
    Tensor encode(const SimplicialComplex &complex) const override;
    Tensor encode(const Diagram &diagram) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const override;
    std::vector<Tensor> encodeBatch(const std::vector<Diagram> &batch_diagrams) const override;
    void setInputSize(Size input_size) override;
    void setOutputSize(Size output_size) override;
    void setParameters(const std::map<std::string, double> &params) override;
    void setEncodingStrategy(const std::string &strategy);
    void setLandscapesParams(Size num_landscapes, Size resolution);
    void setImagesParams(Size resolution, double sigma);
    void setStatisticsParams(bool use_moments, bool use_quantiles);
    Tensor encodeLandscapes(const Diagram &diagram) const;
    Tensor encodeImages(const Diagram &diagram) const;
    Tensor encodeStatistics(const Diagram &diagram) const;
    Tensor encodePersistenceVectors(const Diagram &diagram) const;
    Size getInputSize() const override;
    Size getOutputSize() const override;
    std::string getEncoderType() const override;

private:
    std::string encoding_strategy_;
    Size num_landscapes_;
    Size landscape_resolution_;
    Size image_resolution_;
    double image_sigma_;
    bool use_moments_;
    bool use_quantiles_;
    std::vector<double> computePersistenceLandscapes(const Diagram &diagram, Size num_landscapes,
                                                     Size resolution) const;
    std::vector<double> computePersistenceImages(const Diagram &diagram, Size resolution,
                                                 double sigma) const;
    std::vector<double> computePersistenceStatistics(const Diagram &diagram) const;
    std::vector<double> computePersistenceVectors(const Diagram &diagram) const;
};
class GraphEncoder : public FeatureEncoder
{
public:
    GraphEncoder(Size node_dim, Size edge_dim, Size output_dim);
    Tensor encode(const std::vector<std::vector<double>> &data) const override;
    Tensor encode(const SimplicialComplex &complex) const override;
    Tensor encode(const Diagram &diagram) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const override;
    std::vector<Tensor> encodeBatch(const std::vector<Diagram> &batch_diagrams) const override;
    void setInputSize(Size input_size) override;
    void setOutputSize(Size output_size) override;
    void setParameters(const std::map<std::string, double> &params) override;
    void addGcnLayer(Size hidden_dim);
    void addGatLayer(Size hidden_dim, Size num_heads);
    void addGraphConvLayer(const std::string &conv_type, Size hidden_dim);
    void addGlobalPooling(const std::string &pooling_type);
    void setGraphConstructionMethod(const std::string &method);
    void setDistanceThreshold(double threshold);
    void setKNeighbors(Size k);
    Tensor forward(const Tensor &node_features, const Tensor &edge_features,
                   const Tensor &adjacency) const;
    Size getInputSize() const override;
    Size getOutputSize() const override;
    std::string getEncoderType() const override;

private:
    Size node_dim_;
    Size edge_dim_;
    std::string graph_construction_method_;
    double distance_threshold_;
    Size k_neighbors_;
    struct GCNLayer
    {
        Size input_dim;
        Size output_dim;
        Tensor weights;
        Tensor bias;
    };
    struct GATLayer
    {
        Size input_dim;
        Size output_dim;
        Size num_heads;
        Tensor weights;
        Tensor attention_weights;
    };
    std::vector<GCNLayer> gcn_layers_;
    std::vector<GATLayer> gat_layers_;
    std::string global_pooling_type_;
    Tensor constructGraph(const std::vector<std::vector<double>> &data) const;
    Tensor constructGraphFromComplex(const SimplicialComplex &complex) const;
    Tensor constructGraphFromDiagram(const Diagram &diagram) const;
    Tensor applyGcn(const Tensor &features, const Tensor &adjacency, const GCNLayer &layer) const;
    Tensor applyGat(const Tensor &features, const Tensor &adjacency, const GATLayer &layer) const;
    Tensor applyGlobalPooling(const Tensor &features) const;
};
class HybridEncoder : public FeatureEncoder
{
public:
    HybridEncoder(const std::vector<std::string> &encoder_types);
    Tensor encode(const std::vector<std::vector<double>> &data) const override;
    Tensor encode(const SimplicialComplex &complex) const override;
    Tensor encode(const Diagram &diagram) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<std::vector<std::vector<double>>> &batch_data) const override;
    std::vector<Tensor>
    encodeBatch(const std::vector<SimplicialComplex> &batch_complexes) const override;
    std::vector<Tensor> encodeBatch(const std::vector<Diagram> &batch_diagrams) const override;
    void setInputSize(Size input_size) override;
    void setOutputSize(Size output_size) override;
    void setParameters(const std::map<std::string, double> &params) override;
    void addEncoder(std::unique_ptr<FeatureEncoder> encoder);
    void setFusionMethod(const std::string &fusion_method);
    void setEncoderWeights(const std::vector<double> &weights);
    Tensor concatenateFeatures(const std::vector<Tensor> &features) const;
    Tensor weightedAverageFeatures(const std::vector<Tensor> &features) const;
    Tensor attentionFusion(const std::vector<Tensor> &features) const;
    Size getInputSize() const override;
    Size getOutputSize() const override;
    std::string getEncoderType() const override;

private:
    std::vector<std::unique_ptr<FeatureEncoder>> encoders_;
    std::string fusion_method_;
    std::vector<double> encoder_weights_;
    std::vector<Tensor> encodeWithAllEncoders(const std::vector<std::vector<double>> &data) const;
    std::vector<Tensor> encodeComplexWithAllEncoders(const SimplicialComplex &complex) const;
    std::vector<Tensor> encodeDiagramWithAllEncoders(const Diagram &diagram) const;
};
class EncoderFactory
{
public:
    static std::unique_ptr<FeatureEncoder> createCnnEncoder(Size input_channels,
                                                            Size output_channels,
                                                            const std::vector<Size> &kernel_sizes);
    static std::unique_ptr<FeatureEncoder> createMlpEncoder(Size input_size, Size hidden_size,
                                                            Size output_size, Size num_layers);
    static std::unique_ptr<FeatureEncoder> createTopologicalEncoder(Size feature_dim);
    static std::unique_ptr<FeatureEncoder> createPersistenceEncoder(Size output_dim);
    static std::unique_ptr<FeatureEncoder> createGraphEncoder(Size node_dim, Size edge_dim,
                                                              Size output_dim);
    static std::unique_ptr<FeatureEncoder>
    createHybridEncoder(const std::vector<std::string> &encoder_types);
    static std::unique_ptr<FeatureEncoder> createDefaultPersistenceEncoder();
    static std::unique_ptr<FeatureEncoder> createDefaultComplexEncoder();
    static std::unique_ptr<FeatureEncoder> createDefaultGraphEncoder();
    static std::unique_ptr<FeatureEncoder> createDefaultHybridEncoder();
    static std::unique_ptr<FeatureEncoder> loadEncoderFromConfig(const std::string &config_file);
    static void saveEncoderConfig(const FeatureEncoder &encoder, const std::string &config_file);

private:
    static std::map<std::string, double> loadDefaultParams(const std::string &encoder_type);
    static void validateEncoderConfig(const std::map<std::string, double> &params);
};
class EncoderUtils
{
public:
    static std::vector<std::vector<double>>
    normalizeData(const std::vector<std::vector<double>> &data);
    static std::vector<std::vector<double>>
    standardizeData(const std::vector<std::vector<double>> &data);
    static std::vector<std::vector<double>>
    augmentData(const std::vector<std::vector<double>> &data);
    static std::vector<double> computeFeatureStatistics(const Tensor &features);
    static double computeFeatureDiversity(const std::vector<Tensor> &features);
    static std::vector<double> computeFeatureCorrelation(const Tensor &features1,
                                                         const Tensor &features2);
    static double
    evaluateEncoderPerformance(const FeatureEncoder &encoder,
                               const std::vector<std::vector<std::vector<double>>> &test_data,
                               const std::vector<std::vector<double>> &test_labels);
    static std::vector<double>
    evaluateEncoderRobustness(const FeatureEncoder &encoder,
                              const std::vector<std::vector<std::vector<double>>> &test_data,
                              double noise_level);
    static void visualizeFeatures(const Tensor &features, const std::string &output_file);
    static void
    plotEncoderComparison(const std::vector<std::unique_ptr<FeatureEncoder>> &encoders,
                          const std::vector<std::vector<std::vector<double>>> &test_data,
                          const std::string &output_file);
    static void profileEncoder(const FeatureEncoder &encoder,
                               const std::vector<std::vector<std::vector<double>>> &test_data);
    static std::map<std::string, double>
    benchmarkEncoders(const std::vector<std::unique_ptr<FeatureEncoder>> &encoders,
                      const std::vector<std::vector<std::vector<double>>> &test_data);

private:
    static std::vector<std::vector<double>> applyNoise(const std::vector<std::vector<double>> &data,
                                                       double noise_level);
    static double computeAccuracy(const std::vector<std::vector<double>> &predictions,
                                  const std::vector<std::vector<double>> &labels);
};
} // namespace nerve::encoders
