LossBenchmark benchmarkLoss(int num_dims, int num_pairs)
{
    if (num_dims < 0 || num_pairs < 0)
    {
        throw std::invalid_argument("loss benchmark sizes must be non-negative");
    }

    LossBenchmark bench;
    bench.num_betti_dims = num_dims;
    bench.num_persistence_pairs = num_pairs;

    GPUTopologyLoss::LossConfig config;
    GPUTopologyLoss loss_fn(config);

    std::vector<float> target(num_dims);
    std::vector<float> pred(num_dims);
    std::vector<float> persistence(num_pairs);

    for (int i = 0; i < num_dims; ++i)
    {
        target[i] = static_cast<float>((i * 3) % 10);
        pred[i] = static_cast<float>((i * 7 + 1) % 10);
    }
    for (int i = 0; i < num_pairs; ++i)
    {
        persistence[i] = static_cast<float>((i * 11) % 100);
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    float cpu_loss = 0.0f;
    for (int i = 0; i < num_dims; ++i)
    {
        const float diff = target[i] - pred[i];
        cpu_loss += config.lambda_betti * diff * diff;
    }
    for (float value : persistence)
    {
        cpu_loss += config.lambda_persistence * value;
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    bench.cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    auto start = std::chrono::high_resolution_clock::now();
    float l = loss_fn.combinedLoss(target, pred, persistence);
    auto end = std::chrono::high_resolution_clock::now();
    bench.gpu_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (!std::isfinite(l) || !std::isfinite(cpu_loss))
    {
        throw std::runtime_error("loss benchmark produced a non-finite loss");
    }
    else
    {
        auto ratio = [](double cpu_ms, double gpu_ms) {
            if (!std::isfinite(cpu_ms) || !std::isfinite(gpu_ms) || cpu_ms < 0.0 || gpu_ms <= 0.0)
            {
                return 1.0;
            }
            return cpu_ms / gpu_ms;
        };
        bench.speedup = ratio(bench.cpu_time_ms, bench.gpu_time_ms);
    }

    return bench;
}
