#include "nerve/encoders/encoders.hpp"
#include "nerve/encoders/gpu_encoders.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

int main()
{
    using namespace nerve::encoders;

    CNNEncoder cnn(1, 1, {1});
    MLPEncoder mlp(1, 1, 1, 1);
    GraphEncoder graph(1, 1, 2);
    graph.addGcnLayer(1);
    PersistenceEncoder persistence(2);
    TopologicalEncoder topological(2);

    bool rejected_cnn_weight_overflow = false;
    try
    {
        CNNEncoder huge_cnn(std::numeric_limits<nerve::Size>::max() / 2 + 1, 2, {2});
        (void)huge_cnn;
    }
    catch (const std::length_error &)
    {
        rejected_cnn_weight_overflow = true;
    }
    assert(rejected_cnn_weight_overflow);

    bool rejected_mlp_weight_overflow = false;
    try
    {
        MLPEncoder huge_mlp(std::numeric_limits<nerve::Size>::max() / 2 + 1, 2, 1, 2);
        (void)huge_mlp;
    }
    catch (const std::length_error &)
    {
        rejected_mlp_weight_overflow = true;
    }
    assert(rejected_mlp_weight_overflow);

    bool rejected_cnn_bad_rank = false;
    try
    {
        const Tensor malformed({1.0}, {1});
        (void)cnn.forward(malformed);
    }
    catch (const std::invalid_argument &)
    {
        rejected_cnn_bad_rank = true;
    }
    assert(rejected_cnn_bad_rank);

    bool rejected_cnn_unsafe_forward = false;
    try
    {
        const Tensor unsafe({std::numeric_limits<double>::max()}, {1, 1, 1, 1});
        (void)cnn.forward(unsafe);
    }
    catch (const std::invalid_argument &)
    {
        rejected_cnn_unsafe_forward = true;
    }
    assert(rejected_cnn_unsafe_forward);

    bool rejected_mlp_bad_rank = false;
    try
    {
        const Tensor malformed({1.0}, {1});
        (void)mlp.forward(malformed);
    }
    catch (const std::invalid_argument &)
    {
        rejected_mlp_bad_rank = true;
    }
    assert(rejected_mlp_bad_rank);

    bool rejected_mlp_unsafe_forward = false;
    try
    {
        const Tensor unsafe({std::numeric_limits<double>::max()}, {1, 1});
        (void)mlp.forward(unsafe);
    }
    catch (const std::invalid_argument &)
    {
        rejected_mlp_unsafe_forward = true;
    }
    assert(rejected_mlp_unsafe_forward);

    const Tensor graph_nodes({1.0}, {1, 1});
    const Tensor graph_edges({0.0}, {1});
    const Tensor graph_adjacency({0.0}, {1, 1});
    const auto graph_output = graph.forward(graph_nodes, graph_edges, graph_adjacency);
    assert(graph_output.size() == 2);
    assert(std::all_of(graph_output.data().begin(), graph_output.data().end(),
                       [](double value) { return std::isfinite(value); }));

    bool rejected_graph_weight_overflow = false;
    try
    {
        GraphEncoder huge_graph(std::numeric_limits<nerve::Size>::max() / 2 + 1, 1, 1);
        huge_graph.addGcnLayer(2);
    }
    catch (const std::length_error &)
    {
        rejected_graph_weight_overflow = true;
    }
    assert(rejected_graph_weight_overflow);

    bool rejected_graph_bad_rank = false;
    try
    {
        const Tensor malformed({1.0}, {1});
        (void)graph.forward(malformed, graph_edges, graph_adjacency);
    }
    catch (const std::invalid_argument &)
    {
        rejected_graph_bad_rank = true;
    }
    assert(rejected_graph_bad_rank);

    bool rejected_graph_unsafe_forward = false;
    try
    {
        const Tensor unsafe({std::numeric_limits<double>::max()}, {1, 1});
        (void)graph.forward(unsafe, graph_edges, graph_adjacency);
    }
    catch (const std::invalid_argument &)
    {
        rejected_graph_unsafe_forward = true;
    }
    assert(rejected_graph_unsafe_forward);

    bool rejected_graph_negative_adjacency = false;
    try
    {
        const Tensor negative_adjacency({-1.0}, {1, 1});
        (void)graph.forward(graph_nodes, graph_edges, negative_adjacency);
    }
    catch (const std::invalid_argument &)
    {
        rejected_graph_negative_adjacency = true;
    }
    assert(rejected_graph_negative_adjacency);

    bool rejected_persistence_nonfinite_input = false;
    try
    {
        (void)persistence.encode({{1.0, 2.0, std::numeric_limits<double>::quiet_NaN()}});
    }
    catch (const std::invalid_argument &)
    {
        rejected_persistence_nonfinite_input = true;
    }
    assert(rejected_persistence_nonfinite_input);

    bool rejected_topological_nonfinite_input = false;
    try
    {
        (void)topological.encode({{1.0, 2.0, std::numeric_limits<double>::infinity()}});
    }
    catch (const std::invalid_argument &)
    {
        rejected_topological_nonfinite_input = true;
    }
    assert(rejected_topological_nonfinite_input);

    nerve::persistence::Diagram inverted_diagram;
    inverted_diagram.addPair({1.0, 0.0, 0});
    bool rejected_persistence_invalid_pair = false;
    try
    {
        (void)persistence.encode(inverted_diagram);
    }
    catch (const std::invalid_argument &)
    {
        rejected_persistence_invalid_pair = true;
    }
    assert(rejected_persistence_invalid_pair);

    nerve::persistence::Diagram nan_diagram;
    nan_diagram.addPair({std::numeric_limits<double>::quiet_NaN(), 1.0, 0});
    bool rejected_topological_invalid_pair = false;
    try
    {
        (void)topological.encode(nan_diagram);
    }
    catch (const std::invalid_argument &)
    {
        rejected_topological_invalid_pair = true;
    }
    assert(rejected_topological_invalid_pair);

    nerve::encoders::fusion::FusionConfig fused_config;
    nerve::encoders::fusion::FusedEncoderPipeline fused_pipeline(fused_config);
    std::vector<float> fused_output;
    fused_pipeline.encodeFused({{0.0F, 1.0F}}, fused_output);
    assert(std::all_of(fused_output.begin(), fused_output.end(),
                       [](float value) { return std::isfinite(value); }));

    bool rejected_fused_nan_pair = false;
    try
    {
        fused_pipeline.encodeFused({{0.0F, std::numeric_limits<float>::quiet_NaN()}}, fused_output);
    }
    catch (const std::invalid_argument &)
    {
        rejected_fused_nan_pair = true;
    }
    assert(rejected_fused_nan_pair);

    nerve::encoders::fusion::FusionConfig unfused_config;
    unfused_config.fuse_persistence = false;
    nerve::encoders::fusion::FusedEncoderPipeline unfused_pipeline(unfused_config);
    bool rejected_unfused_inverted_pair = false;
    try
    {
        unfused_pipeline.encodeFused({{1.0F, 0.0F}}, fused_output);
    }
    catch (const std::invalid_argument &)
    {
        rejected_unfused_inverted_pair = true;
    }
    assert(rejected_unfused_inverted_pair);

    nerve::encoders::fusion::AsyncEncoderExecutor async_encoder(1);
    async_encoder.submit({{0.0F, -std::numeric_limits<float>::infinity()}});
    bool rejected_async_invalid_pair = false;
    try
    {
        (void)async_encoder.flush();
    }
    catch (const std::invalid_argument &)
    {
        rejected_async_invalid_pair = true;
    }
    assert(rejected_async_invalid_pair);

    bool rejected_normalize_overflow = false;
    try
    {
        (void)EncoderUtils::normalizeData(
            {{-std::numeric_limits<double>::max()}, {std::numeric_limits<double>::max()}});
    }
    catch (const std::overflow_error &)
    {
        rejected_normalize_overflow = true;
    }
    assert(rejected_normalize_overflow);

    bool rejected_standardize_overflow = false;
    try
    {
        (void)EncoderUtils::standardizeData({{std::numeric_limits<double>::max()}, {0.0}});
    }
    catch (const std::overflow_error &)
    {
        rejected_standardize_overflow = true;
    }
    assert(rejected_standardize_overflow);

    bool rejected_statistics_overflow = false;
    try
    {
        const Tensor huge_features({std::numeric_limits<double>::max(), 0.0}, {2});
        (void)EncoderUtils::computeFeatureStatistics(huge_features);
    }
    catch (const std::overflow_error &)
    {
        rejected_statistics_overflow = true;
    }
    assert(rejected_statistics_overflow);

    bool rejected_diversity_overflow = false;
    try
    {
        const Tensor first({std::numeric_limits<double>::max()}, {1});
        const Tensor second({0.0}, {1});
        (void)EncoderUtils::computeFeatureDiversity({first, second});
    }
    catch (const std::overflow_error &)
    {
        rejected_diversity_overflow = true;
    }
    assert(rejected_diversity_overflow);

    return 0;
}
