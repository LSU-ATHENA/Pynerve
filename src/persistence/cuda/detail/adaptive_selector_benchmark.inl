SelectorBenchmark benchmarkSelector(int num_pairs, int num_algorithms)
{
    if (num_pairs < 0 || num_algorithms <= 0)
    {
        throw std::invalid_argument(
            "selector benchmark sizes must be non-negative with algorithms");
    }

    SelectorBenchmark bench{};
    bench.num_pairs = num_pairs;
    bench.num_algorithms = num_algorithms;

    SelectorConfig config;
    config.num_algorithms = num_algorithms;
    config.feature_dim = 20;
    config.hidden_dim = 64;

    GPUAdaptiveSelector selector(config);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> birth_dist(0.0f, 100.0f);
    std::uniform_real_distribution<float> lifetime_dist(1.0f, 100.0f);
    std::uniform_int_distribution<int> dim_dist(0, 3);
    std::vector<std::pair<float, float>> pairs;
    std::vector<int> dims;
    pairs.reserve(static_cast<size_t>(num_pairs));
    dims.reserve(static_cast<size_t>(num_pairs));
    for (int i = 0; i < num_pairs; ++i)
    {
        const float birth = birth_dist(gen);
        pairs.emplace_back(birth, birth + lifetime_dist(gen));
        dims.push_back(dim_dist(gen));
    }

    auto start_feature = std::chrono::high_resolution_clock::now();
    selector.extractFeaturesFromPersistence(pairs, dims);
    auto end_feature = std::chrono::high_resolution_clock::now();
    bench.gpu_feature_ms =
        std::chrono::duration<double, std::milli>(end_feature - start_feature).count();

    auto start_predict = std::chrono::high_resolution_clock::now();
    (void)selector.predict();
    auto end_predict = std::chrono::high_resolution_clock::now();
    bench.gpu_predict_ms =
        std::chrono::duration<double, std::milli>(end_predict - start_predict).count();

    bench.cpu_feature_ms = 0.0;
    bench.cpu_predict_ms = 0.0;
    bench.speedup_feature = 1.0;
    bench.speedup_predict = 1.0;
    return bench;
}
