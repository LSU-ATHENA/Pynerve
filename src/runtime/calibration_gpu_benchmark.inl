CalibrationBenchmark benchmarkCalibration(int num_samples, int num_buckets)
{
    if (num_samples < 0 || num_buckets <= 0)
    {
        throw std::invalid_argument("calibration benchmark dimensions are invalid");
    }
    CalibrationBenchmark bench;
    bench.num_samples = num_samples;
    bench.num_buckets = num_buckets;

    GPUCalibrationModel model(num_buckets);

    std::vector<GPUCalibrationModel::Sample> samples;
    samples.reserve(std::max(0, num_samples));
    for (int i = 0; i < num_samples; ++i)
    {
        samples.push_back({"hw_" + std::to_string(i % 10), "prob_" + std::to_string(i % 20),
                           "alg_" + std::to_string(i % 5), static_cast<double>((i * 37) % 1000),
                           static_cast<double>(((i * 37) + (i % 11)) % 1000), 0.8,
                           static_cast<double>(i)});
    }

    struct CpuBucket
    {
        double predicted_sum = 0.0;
        double observed_sum = 0.0;
        double error_sum = 0.0;
        double confidence = 0.0;
        double error_bound = 1.0;
        int count = 0;
    };
    auto bucket_id = [num_buckets](const auto &s) {
        std::hash<std::string> hasher;
        const size_t h =
            hasher(s.hardware_fingerprint) ^ hasher(s.problem_bucket) ^ hasher(s.algorithm);
        return static_cast<int>(h % std::max(1, num_buckets));
    };

    std::vector<CpuBucket> cpu_buckets(std::max(1, num_buckets));
    auto start_cpu_update = std::chrono::high_resolution_clock::now();
    for (const auto &sample : samples)
    {
        CpuBucket &bucket = cpu_buckets[bucket_id(sample)];
        bucket.predicted_sum += sample.predicted_time_ms;
        bucket.observed_sum += sample.observed_time_ms;
        bucket.error_sum += std::abs(sample.predicted_time_ms - sample.observed_time_ms);
        bucket.count += 1;
    }
    for (auto &bucket : cpu_buckets)
    {
        if (bucket.count < 20)
        {
            continue;
        }
        const double avg_error = bucket.error_sum / bucket.count;
        const double sample_factor = static_cast<double>(bucket.count) / (bucket.count + 5);
        bucket.confidence = std::clamp((1.0 - avg_error) * sample_factor, 0.0, 1.0);
        bucket.error_bound = avg_error + 1.28 * (avg_error * 1.25);
    }
    auto end_cpu_update = std::chrono::high_resolution_clock::now();
    bench.cpu_update_ms =
        std::chrono::duration<double, std::milli>(end_cpu_update - start_cpu_update).count();

    auto start_update = std::chrono::high_resolution_clock::now();
    model.addSamples(samples);
    auto end_update = std::chrono::high_resolution_clock::now();
    bench.gpu_update_ms =
        std::chrono::duration<double, std::milli>(end_update - start_update).count();

    auto start_cpu_query = std::chrono::high_resolution_clock::now();
    const GPUCalibrationModel::Sample query_sample{"hw_0", "prob_0", "alg_0", 0.0, 0.0, 0.0, 0.0};
    const CpuBucket &cpu_bucket = cpu_buckets[bucket_id(query_sample)];
    volatile double cpu_query_observed =
        cpu_bucket.count > 0 ? cpu_bucket.observed_sum / cpu_bucket.count : 0.0;
    (void)cpu_query_observed;
    auto end_cpu_query = std::chrono::high_resolution_clock::now();
    bench.cpu_query_ms =
        std::chrono::duration<double, std::milli>(end_cpu_query - start_cpu_query).count();

    auto start_query = std::chrono::high_resolution_clock::now();
    auto stats = model.queryBucket("hw_0", "prob_0", "alg_0");
    auto end_query = std::chrono::high_resolution_clock::now();
    bench.gpu_query_ms = std::chrono::duration<double, std::milli>(end_query - start_query).count();
    if (stats.sample_count < 0)
    {
        throw std::runtime_error("calibration benchmark returned a negative sample count");
    }

    const int feature_count = 4;
    const int fit_samples = std::max(1, std::min(num_samples, 4096));
    std::vector<std::vector<float>> features(fit_samples, std::vector<float>(feature_count));
    std::vector<float> targets(fit_samples);
    for (int i = 0; i < fit_samples; ++i)
    {
        for (int f = 0; f < feature_count; ++f)
        {
            features[i][f] = static_cast<float>(((i + 1) * (f + 3)) % 17) / 17.0f;
        }
        targets[i] = 0.25f * features[i][0] + 0.5f * features[i][1] + 0.125f;
    }

    auto start_cpu_fit = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_weights(feature_count, 0.0f);
    float cpu_bias = 0.0f;
    for (int epoch = 0; epoch < 100; ++epoch)
    {
        std::vector<float> gradients(feature_count, 0.0f);
        float bias_grad = 0.0f;
        for (int i = 0; i < fit_samples; ++i)
        {
            float prediction = cpu_bias;
            for (int f = 0; f < feature_count; ++f)
            {
                prediction += cpu_weights[f] * features[i][f];
            }
            const float error = prediction - targets[i];
            bias_grad += error;
            for (int f = 0; f < feature_count; ++f)
            {
                gradients[f] += error * features[i][f];
            }
        }
        for (int f = 0; f < feature_count; ++f)
        {
            cpu_weights[f] -= 0.01f * ((gradients[f] / fit_samples) + 0.001f * cpu_weights[f]);
        }
        cpu_bias -= 0.01f * (bias_grad / fit_samples);
    }
    auto end_cpu_fit = std::chrono::high_resolution_clock::now();
    bench.cpu_fit_ms =
        std::chrono::duration<double, std::milli>(end_cpu_fit - start_cpu_fit).count();

    auto start_gpu_fit = std::chrono::high_resolution_clock::now();
    model.fitLinearModel(features, targets, 100, 0.01f);
    auto end_gpu_fit = std::chrono::high_resolution_clock::now();
    bench.gpu_fit_ms =
        std::chrono::duration<double, std::milli>(end_gpu_fit - start_gpu_fit).count();

    auto ratio = [](double cpu_ms, double gpu_ms) {
        if (!std::isfinite(cpu_ms) || !std::isfinite(gpu_ms) || cpu_ms < 0.0 || gpu_ms < 0.0 ||
            gpu_ms == 0.0)
        {
            return 1.0;
        }
        return cpu_ms / gpu_ms;
    };
    bench.speedup_update = ratio(bench.cpu_update_ms, bench.gpu_update_ms);
    bench.speedup_query = ratio(bench.cpu_query_ms, bench.gpu_query_ms);
    bench.speedup_fit = ratio(bench.cpu_fit_ms, bench.gpu_fit_ms);

    return bench;
}
