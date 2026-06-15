
#pragma once

#include <future>
#include <memory>
#include <utility>
#include <vector>

namespace nerve::encoders
{
namespace gpu
{

struct EncoderGPUBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double fused_time_ms;
    double speedup_gpu;
    double speedup_fused;
    int batch_size;
    int num_features;
};

EncoderGPUBenchmark benchmarkGPUEncoder(int batch_size, int num_features);

} // namespace gpu

namespace tensorcore
{

class TensorCoreMLPEncoder
{
public:
    TensorCoreMLPEncoder(int input_dim, const std::vector<int> &hidden_dims, int output_dim);
    ~TensorCoreMLPEncoder();

    void encode(const std::vector<float> &input, std::vector<float> &output, int batch_size);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class CUDNNTopologicalEncoder
{
public:
    CUDNNTopologicalEncoder(int input_height, int input_width, int input_channels, int num_filters,
                            int filter_size);
    ~CUDNNTopologicalEncoder();

    void encode(const std::vector<float> &input, std::vector<float> &output);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class MixedPrecisionEncoder
{
public:
    MixedPrecisionEncoder();

    float scaleLoss(float loss);
    void unscaleGradients(float *gradients, int n);
    bool checkForInfNan(const float *gradients, int n);
    void updateLossScale(bool found_inf);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct TensorCoreBenchmark
{
    double fp32_time_ms;
    double fp16_time_ms;
    double speedup;
    int batch_size;
    int input_dim;
    int output_dim;
};

TensorCoreBenchmark benchmarkTensorCore(int batch_size, int input_dim, int output_dim);

} // namespace tensorcore

namespace fusion
{

struct FusionConfig
{
    bool fuse_persistence = true;
    bool fuse_normalization = true;
    bool fuse_activation = true;
    bool use_persistent_kernels = false;
};

class FusedEncoderPipeline
{
public:
    explicit FusedEncoderPipeline(const FusionConfig &config);
    ~FusedEncoderPipeline();

    void encodeFused(const std::vector<std::pair<float, float>> &diagram,
                     std::vector<float> &output);

    void encodePersistent(const std::vector<std::vector<std::pair<float, float>>> &diagrams,
                          std::vector<std::vector<float>> &outputs);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class AsyncEncoderExecutor
{
public:
    explicit AsyncEncoderExecutor(size_t batch_size = 32);
    ~AsyncEncoderExecutor();

    void submit(const std::vector<std::pair<float, float>> &diagram);
    std::vector<std::vector<float>> flush();

    std::future<std::vector<float>>
    submitAsync(const std::vector<std::pair<float, float>> &diagram);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class MemoryOptimizedEncoder
{
public:
    struct MemoryConfig
    {
        bool enable_checkpointing = true;
        bool activation_compression = true;
        int compression_bits = 8;
        size_t max_memory_mb = 1024;
    };

    explicit MemoryOptimizedEncoder(const MemoryConfig &config);
    ~MemoryOptimizedEncoder();

    std::vector<float> encodeWithCheckpointing(const std::vector<float> &input);
    std::vector<float> backward(const std::vector<float> &grad_output);
    void clearCheckpoints();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct FusionBenchmark
{
    double unfused_time_ms;
    double fused_time_ms;
    double speedup;
    int num_diagrams;
    int features_per_diagram;
};

FusionBenchmark benchmarkFusedEncoder(int num_diagrams, int features_per_diagram);

} // namespace fusion
} // namespace nerve::encoders
