
#pragma once

#include <vector>

namespace nerve::compression
{
namespace gpu
{

struct AutoencoderBenchmark
{
    double cpu_encode_ms;
    double gpu_encode_ms;
    double cpu_decode_ms;
    double gpu_decode_ms;
    double cpu_roundtrip_ms;
    double gpu_roundtrip_ms;
    double speedup_encode;
    double speedup_decode;
    int input_dim;
    int latent_dim;
    int batch_size;
};

AutoencoderBenchmark benchmarkAutoencoder(int input_dim, int latent_dim, int batch_size);

struct CuDNNEncoderBenchmark
{
    double cpu_time_ms;
    double cudnn_time_ms;
    double fused_time_ms;
    double speedup_cudnn;
    double speedup_fused;
    int batch_size;
    int input_dim;
    int latent_dim;
};

CuDNNEncoderBenchmark benchmarkCuDNNEncoder(int batch_size, int input_dim, int latent_dim);

struct CuDNNDecoderBenchmark
{
    double cpu_time_ms;
    double cudnn_time_ms;
    double fused_time_ms;
    double speedup_cudnn;
    double speedup_fused;
    int batch_size;
    int latent_dim;
    int output_dim;
};

CuDNNDecoderBenchmark benchmarkCuDNNDecoder(int batch_size, int latent_dim, int output_dim);

} // namespace gpu
} // namespace nerve::compression
