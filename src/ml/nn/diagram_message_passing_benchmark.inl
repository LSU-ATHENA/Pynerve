std::vector<float> cpuAttentionForward(const std::vector<float> &features,
                                       const std::vector<float> &diagrams, int num_pairs,
                                       int feature_dim, float temperature)
{
    std::vector<float> output(num_pairs * feature_dim, 0.0f);
    std::vector<float> scores(num_pairs, 0.0f);
    const float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(feature_dim));

    for (int pair = 0; pair < num_pairs; ++pair)
    {
        for (int neighbor = 0; neighbor < num_pairs; ++neighbor)
        {
            float dot = 0.0f;
            for (int feat = 0; feat < feature_dim; ++feat)
            {
                dot +=
                    features[pair * feature_dim + feat] * features[neighbor * feature_dim + feat];
            }

            const float b1 = diagrams[pair * DIAGRAM_FIELDS_PER_PAIR + 0];
            const float d1 = diagrams[pair * DIAGRAM_FIELDS_PER_PAIR + 1];
            const float b2 = diagrams[neighbor * DIAGRAM_FIELDS_PER_PAIR + 0];
            const float d2 = diagrams[neighbor * DIAGRAM_FIELDS_PER_PAIR + 1];
            const float db = b1 - b2;
            const float dd = d1 - d2;
            scores[neighbor] =
                dot * inv_sqrt_dim - std::sqrt(db * db + dd * dd) * TOPOLOGICAL_BIAS_PENALTY;
        }

        const float max_score = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        for (float &score : scores)
        {
            score = std::exp((score - max_score) / temperature);
            sum_exp += score;
        }

        const float inv_sum = 1.0f / std::max(sum_exp, EPSILON);
        for (int neighbor = 0; neighbor < num_pairs; ++neighbor)
        {
            const float weight = scores[neighbor] * inv_sum;
            for (int feat = 0; feat < feature_dim; ++feat)
            {
                output[pair * feature_dim + feat] +=
                    weight * features[neighbor * feature_dim + feat];
            }
        }
    }

    return output;
}

/**
 * @brief Benchmark diagram GNN
 */
struct DiagramGNNSpeedup
{
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    int num_pairs;
    int feature_dim;
};

DiagramGNNSpeedup benchmarkDiagramGNN(int num_pairs, int feature_dim)
{
    DiagramGNNSpeedup bench;
    bench.num_pairs = num_pairs;
    bench.feature_dim = feature_dim;

    GPUDiagramGNNLayer gnn(num_pairs, feature_dim, GPUDiagramGNNLayer::AggregationType::ATTENTION);

    // Deterministic diagram workload for CPU/GPU timing.
    std::vector<float> diagrams(num_pairs * DIAGRAM_FIELDS_PER_PAIR);
    for (int i = 0; i < num_pairs; ++i)
    {
        diagrams[i * DIAGRAM_FIELDS_PER_PAIR + 0] = static_cast<float>((i * 11) % 97) / 97.0f;
        diagrams[i * DIAGRAM_FIELDS_PER_PAIR + 1] = diagrams[i * DIAGRAM_FIELDS_PER_PAIR + 0] +
                                                    0.05f +
                                                    static_cast<float>((i * 7) % 31) / 310.0f;
        diagrams[i * DIAGRAM_FIELDS_PER_PAIR + 2] = static_cast<float>(i % DIAGRAM_FIELDS_PER_PAIR);
    }

    std::vector<float> features(num_pairs * feature_dim);
    for (size_t i = 0; i < features.size(); ++i)
    {
        features[i] = static_cast<float>((i * 13) % 101) / 101.0f;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_output = cpuAttentionForward(features, diagrams, num_pairs, feature_dim, 1.0f);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    volatile float cpu_sink = cpu_output.empty() ? 0.0f : cpu_output.back();
    (void)cpu_sink;

    // Compute adjacency
    gnn.computeAdjacency(diagrams, DEFAULT_BANDWIDTH);

    // GPU benchmark
    auto start = std::chrono::high_resolution_clock::now();
    auto output = gnn.forward(features, diagrams);
    auto end = std::chrono::high_resolution_clock::now();
    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    bench.speedup = bench.gpu_time_ms > 0.0 ? bench.cpu_time_ms / bench.gpu_time_ms : 0.0;

    return bench;
}
