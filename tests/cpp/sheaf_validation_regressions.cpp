#include "nerve/config.hpp"
#include "nerve/sheaf/sheaf_laplacian.hpp"
#include "nerve/sheaf/sheaf_learning.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
#if HAS_EIGEN && __has_include(<Eigen/Sparse>) && __has_include(<Eigen/Dense>)
    using Node = nerve::sheaf::SheafLaplacianRuntime::SheafNode;

    Node node;
    node.node_id = 1;
    node.position = {0.0, 1.0};
    node.attributes = {2.0};
    node.attribute_names = {"signal"};
    node.neighbors = {2};
    node.edge_weights = {1.0};
    node.stalk_values = {3.0};
    node.restriction_maps = {1.0};
    node.local_laplacian = 0.5;
    assert(node.isValid());

    Node invalid = node;
    invalid.position[0] = std::numeric_limits<double>::quiet_NaN();
    assert(!invalid.isValid());

    invalid = node;
    invalid.attributes[0] = std::numeric_limits<double>::infinity();
    assert(!invalid.isValid());

    invalid = node;
    invalid.edge_weights[0] = std::numeric_limits<double>::quiet_NaN();
    assert(!invalid.isValid());

    invalid = node;
    invalid.stalk_values[0] = std::numeric_limits<double>::infinity();
    assert(!invalid.isValid());

    invalid = node;
    invalid.restriction_maps[0] = std::numeric_limits<double>::quiet_NaN();
    assert(!invalid.isValid());

    invalid = node;
    invalid.local_laplacian = std::numeric_limits<double>::infinity();
    assert(!invalid.isValid());

    nerve::sheaf::SheafLaplacianRuntime::SheafConfig config;
    nerve::sheaf::SheafLaplacianRuntime runtime(config);
    auto invalid_config = config;
    invalid_config.attribute_weight = std::numeric_limits<double>::quiet_NaN();
    bool rejected_invalid_constructor_config = false;
    try {
        nerve::sheaf::SheafLaplacianRuntime invalid_runtime(invalid_config);
        (void)invalid_runtime;
    } catch (const std::invalid_argument&) {
        rejected_invalid_constructor_config = true;
    }
    assert(rejected_invalid_constructor_config);

    invalid_config = config;
    invalid_config.topological_weight = std::numeric_limits<double>::infinity();
    bool rejected_invalid_set_config = false;
    try {
        runtime.setConfig(invalid_config);
    } catch (const std::invalid_argument&) {
        rejected_invalid_set_config = true;
    }
    assert(rejected_invalid_set_config);

    nerve::sheaf::learning::SheafLearningConfig learning_config;
    auto invalid_learning_config = learning_config;
    invalid_learning_config.learning_rate = std::numeric_limits<double>::quiet_NaN();
    bool rejected_invalid_learning_constructor_config = false;
    try {
        nerve::sheaf::learning::SheafLearner invalid_learner(invalid_learning_config);
        (void)invalid_learner;
    } catch (const std::invalid_argument&) {
        rejected_invalid_learning_constructor_config = true;
    }
    assert(rejected_invalid_learning_constructor_config);

    nerve::sheaf::learning::SheafLearner learner(learning_config);
    invalid_learning_config = learning_config;
    invalid_learning_config.convergence_tolerance = std::numeric_limits<double>::infinity();
    bool rejected_invalid_learning_set_config = false;
    try {
        learner.setConfig(invalid_learning_config);
    } catch (const std::invalid_argument&) {
        rejected_invalid_learning_set_config = true;
    }
    assert(rejected_invalid_learning_set_config);

    Eigen::SparseMatrix<double> adjacency(2, 2);
    std::vector<Eigen::Triplet<double>> adjacency_triplets{
        Eigen::Triplet<double>(0, 1, 1.0), Eigen::Triplet<double>(1, 0, 1.0)};
    adjacency.setFromTriplets(adjacency_triplets.begin(), adjacency_triplets.end());
    std::vector<Eigen::MatrixXd> node_data(2, Eigen::MatrixXd::Ones(1, 2));
    std::vector<size_t> node_dimensions{1, 1};
    assert(learner.learnSheafClosedForm(adjacency, node_data, node_dimensions).isValid());

    auto invalid_adjacency = adjacency;
    invalid_adjacency.coeffRef(0, 1) = std::numeric_limits<double>::infinity();
    bool rejected_nonfinite_learning_graph = false;
    try {
        (void)learner.learnSheafClosedForm(invalid_adjacency, node_data, node_dimensions);
    } catch (const std::invalid_argument&) {
        rejected_nonfinite_learning_graph = true;
    }
    assert(rejected_nonfinite_learning_graph);

    auto invalid_node_data = node_data;
    invalid_node_data[1](0, 0) = std::numeric_limits<double>::quiet_NaN();
    bool rejected_nonfinite_learning_data = false;
    try {
        (void)learner.learnSheafClosedForm(adjacency, invalid_node_data, node_dimensions);
    } catch (const std::invalid_argument&) {
        rejected_nonfinite_learning_data = true;
    }
    assert(rejected_nonfinite_learning_data);

    bool rejected_nonfinite_graph_signal = false;
    try {
        const std::vector<std::pair<uint32_t, uint32_t>> graph_edges{{0, 1}};
        const std::vector<std::vector<double>> node_signals{
            {0.0, 1.0}, {std::numeric_limits<double>::quiet_NaN(), 2.0}};
        (void)nerve::sheaf::learning::learnSheafFromGraphSignal(graph_edges, node_signals);
    } catch (const std::invalid_argument&) {
        rejected_nonfinite_graph_signal = true;
    }
    assert(rejected_nonfinite_graph_signal);

    nerve::sheaf::learning::SheafLearningResult result;
    result.learned_sheaf_laplacian.resize(2, 2);
    result.learned_sheaf_laplacian.insert(0, 0) = 1.0;
    result.restriction_maps = {Eigen::MatrixXd::Identity(1, 1)};
    result.final_total_variation = 0.0;
    result.iterations_used = 1;
    result.converged = true;
    result.compute_time_ms = 0.0;
    assert(result.isValid());

    auto invalid_result = result;
    invalid_result.compute_time_ms = std::numeric_limits<double>::quiet_NaN();
    assert(!invalid_result.isValid());

    invalid_result = result;
    invalid_result.learned_sheaf_laplacian.coeffRef(0, 0) =
        std::numeric_limits<double>::infinity();
    assert(!invalid_result.isValid());

    invalid_result = result;
    invalid_result.restriction_maps[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
    assert(!invalid_result.isValid());

    nerve::sheaf::SheafLaplacianRuntime::SheafLaplacianResult sheaf_result;
    sheaf_result.sheaf_laplacian.resize(2, 2);
    sheaf_result.sheaf_laplacian.insert(0, 0) = 1.0;
    sheaf_result.sheaf_laplacian.insert(1, 1) = 1.0;
    sheaf_result.topological_laplacian.resize(1, 1);
    sheaf_result.topological_laplacian.insert(0, 0) = 1.0;
    sheaf_result.attribute_laplacian.resize(1, 1);
    sheaf_result.attribute_laplacian.insert(0, 0) = 2.0;
    sheaf_result.attribute_names = {"signal"};
    assert(nerve::sheaf::sheaf_utils::validateSheafProperties(sheaf_result));
    assert(std::isfinite(nerve::sheaf::sheaf_utils::computeAttributeInfluence(sheaf_result)));
    assert(std::isfinite(
        nerve::sheaf::sheaf_utils::computeTopologicalInfluence(sheaf_result)));

    auto invalid_sheaf_result = sheaf_result;
    invalid_sheaf_result.sheaf_laplacian.coeffRef(0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    assert(!nerve::sheaf::sheaf_utils::validateSheafProperties(invalid_sheaf_result));

    invalid_sheaf_result = sheaf_result;
    invalid_sheaf_result.sheaf_laplacian.coeffRef(0, 0) = -1.0;
    assert(!nerve::sheaf::sheaf_utils::validateSheafProperties(invalid_sheaf_result));

    invalid_sheaf_result = sheaf_result;
    invalid_sheaf_result.attribute_laplacian.coeffRef(0, 0) =
        std::numeric_limits<double>::infinity();
    assert(!nerve::sheaf::sheaf_utils::validateSheafProperties(invalid_sheaf_result));
    assert(std::isfinite(
        nerve::sheaf::sheaf_utils::computeAttributeInfluence(invalid_sheaf_result)));
    assert(std::isfinite(
        nerve::sheaf::sheaf_utils::computeTopologicalInfluence(invalid_sheaf_result)));
    const auto contributions =
        nerve::sheaf::sheaf_utils::computePerAttributeContributions(invalid_sheaf_result);
    assert(contributions.size() == 1);
    assert(std::isfinite(contributions[0]));
#endif

    return 0;
}
