AutoencoderBenchmark benchmarkAutoencoder(int input_dim, int latent_dim, int batch_size)
{
    AutoencoderBenchmark bench;
    bench.input_dim = input_dim;
    bench.latent_dim = latent_dim;
    bench.batch_size = batch_size;

    const int hidden_dim = std::max(1, input_dim / 2);
    BenchmarkAutoencoder::Config config;
    config.encoder_layers = {input_dim, hidden_dim, latent_dim};
    config.decoder_layers = {latent_dim, hidden_dim, input_dim};
    config.batch_size = batch_size;

    BenchmarkAutoencoder autoencoder(config);

    std::vector<float> input(batch_size * input_dim);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<float>((i * 17) % 251) / 251.0f;
    }

    auto cpu_forward = [](const std::vector<float> &input_data, int batch,
                          const std::vector<int> &layers, bool sigmoid_output) {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::vector<float> current = input_data;
        for (size_t layer = 0; layer + 1 < layers.size(); ++layer)
        {
            const int in = layers[layer];
            const int out = layers[layer + 1];
            std::vector<float> weights(static_cast<size_t>(in) * out);
            for (auto &w : weights)
            {
                w = dist(gen);
            }
            std::vector<float> next(static_cast<size_t>(batch) * out, 0.0f);
            for (int row = 0; row < batch; ++row)
            {
                for (int col = 0; col < out; ++col)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < in; ++k)
                    {
                        sum += current[static_cast<size_t>(row) * in + k] *
                               weights[static_cast<size_t>(col) * in + k];
                    }
                    const bool final_layer = layer + 2 == layers.size();
                    if (!final_layer || !sigmoid_output)
                    {
                        sum = std::max(0.0f, sum);
                    }
                    else
                    {
                        sum = 1.0f / (1.0f + std::exp(-sum));
                    }
                    next[static_cast<size_t>(row) * out + col] = sum;
                }
            }
            current = std::move(next);
        }
        return current;
    };

    auto start_cpu_encode = std::chrono::high_resolution_clock::now();
    auto cpu_latent = cpu_forward(input, batch_size, config.encoder_layers, false);
    auto end_cpu_encode = std::chrono::high_resolution_clock::now();
    bench.cpu_encode_ms =
        std::chrono::duration<double, std::milli>(end_cpu_encode - start_cpu_encode).count();

    auto start_encode = std::chrono::high_resolution_clock::now();
    auto latent = autoencoder.encode(input);
    auto end_encode = std::chrono::high_resolution_clock::now();
    bench.gpu_encode_ms =
        std::chrono::duration<double, std::milli>(end_encode - start_encode).count();

    auto start_decode = std::chrono::high_resolution_clock::now();
    auto output = autoencoder.decode(latent);
    auto end_decode = std::chrono::high_resolution_clock::now();
    bench.gpu_decode_ms =
        std::chrono::duration<double, std::milli>(end_decode - start_decode).count();

    auto start_cpu_decode = std::chrono::high_resolution_clock::now();
    auto cpu_output = cpu_forward(cpu_latent, batch_size, config.decoder_layers, true);
    auto end_cpu_decode = std::chrono::high_resolution_clock::now();
    bench.cpu_decode_ms =
        std::chrono::duration<double, std::milli>(end_cpu_decode - start_cpu_decode).count();
    bench.cpu_roundtrip_ms = bench.cpu_encode_ms + bench.cpu_decode_ms;
    bench.gpu_roundtrip_ms = bench.gpu_encode_ms + bench.gpu_decode_ms;

    auto ratio = [](double cpu_ms, double gpu_ms) {
        if (!std::isfinite(cpu_ms) || !std::isfinite(gpu_ms) || cpu_ms < 0.0 || gpu_ms < 0.0 ||
            gpu_ms == 0.0)
        {
            return 1.0;
        }
        return cpu_ms / gpu_ms;
    };
    bench.speedup_encode = ratio(bench.cpu_encode_ms, bench.gpu_encode_ms);
    bench.speedup_decode = ratio(bench.cpu_decode_ms, bench.gpu_decode_ms);
    bench.speedup_fused = 1.0;
    if (cpu_output.empty() || output.empty())
    {
        throw std::runtime_error("autoencoder benchmark produced empty output");
    }

    return bench;
}
