#pragma once

// C++20 Diagram Convolution for Neural Networks
// Replaces: python/nerve/nn/diagram_conv.py
// Provides: High-performance diagram convolution with GPU support

#include <memory>
#include <span>
#include <vector>

namespace nerve::nn
{

// Diagram Convolution 1D

template <typename T = float>
class DiagramConv1D
{
public:
    struct Config
    {
        int in_channels = 1;
        int out_channels = 1;
        int kernel_size = 5;
        int stride = 1;
        bool use_persistence_weighting = true;
    };

    struct Weights
    {
        std::vector<T> kernel; // [out_channels, in_channels * 2, kernel_size]
        std::vector<T> bias;   // [out_channels]
    };

    explicit DiagramConv1D(Config config);

    // Input: [batch, n_pairs, 3] (birth, death, dim) + [batch, n_pairs,
    // in_channels] features Output: [batch, out_channels, n_pairs]
    [[nodiscard]] std::vector<T>
    forward(std::span<const T> diagram,  // [batch, n_pairs, 3]
            std::span<const T> features, // [batch, n_pairs, in_channels]
            size_t batch_size, size_t n_pairs) const;

    // Set weights from external source (e.g., PyTorch)
    void set_weights(std::span<const T> kernel, std::span<const T> bias);

    // Get output size
    [[nodiscard]] size_t output_size(size_t n_pairs) const;

private:
    Config config_;
    Weights weights_;

    [[nodiscard]] std::vector<T> conv1d(std::span<const T> input, size_t batch_size,
                                        size_t n_pairs) const;

    // Persistence-based gating
    [[nodiscard]] std::vector<T> apply_persistence_gate(std::span<const T> output,
                                                        std::span<const T> diagram,
                                                        size_t batch_size, size_t n_pairs) const;
};

// Diagram Convolution 2D (on birth-death plane)

template <typename T = float>
class DiagramConv2D
{
public:
    struct Config
    {
        int in_channels = 1;
        int out_channels = 1;
        int kernel_h = 3;
        int kernel_w = 3;
        int stride_h = 1;
        int stride_w = 1;
    };

    explicit DiagramConv2D(Config config);

    // Forward on 2D birth-death grid
    // Input: [batch, in_channels, height, width]
    // Output: [batch, out_channels, out_h, out_w]
    [[nodiscard]] std::vector<T> forward(std::span<const T> input, size_t batch_size, size_t height,
                                         size_t width) const;

private:
    Config config_;
    std::vector<T> kernel_; // [out_channels, in_channels, kernel_h, kernel_w]
    std::vector<T> bias_;   // [out_channels]
};

// Persistence Image Layer (Convert diagram to 2D image)

template <typename T = float>
class PersistenceImageLayer
{
public:
    struct Config
    {
        int resolution_h = 20;
        int resolution_w = 20;
        T sigma = T(0.1); // Gaussian kernel sigma
        enum class Weight
        {
            LINEAR,
            QUADRATIC,
            CONSTANT
        } weight = Weight::LINEAR;
    };

    explicit PersistenceImageLayer(Config config);

    // Convert persistence diagram to image
    // Input: [batch, n_pairs, 3] (birth, death, dim)
    // Output: [batch, resolution_h, resolution_w]
    [[nodiscard]] std::vector<T> forward(std::span<const T> diagram, size_t batch_size,
                                         size_t n_pairs) const;

    // Multi-dimension: separate image per dimension
    [[nodiscard]] std::vector<T> forward_multi_dim(std::span<const T> diagram, size_t batch_size,
                                                   size_t n_pairs, int max_dim) const;

private:
    Config config_;

    [[nodiscard]] std::vector<T> gaussian_kernel_2d(T sigma, int size) const;
    [[nodiscard]] T compute_weight(T persistence) const;
};

// Landscape Layer (Convert diagram to Betti curve)

template <typename T = float>
class LandscapeLayer
{
public:
    struct Config
    {
        int n_layers = 5;     // Number of landscape functions
        int resolution = 100; // Resolution along persistence axis
        T min_persistence = T(0.0);
    };

    explicit LandscapeLayer(Config config);

    // Convert diagram to persistence landscape
    // Input: [batch, n_pairs, 3]
    // Output: [batch, n_layers, resolution]
    [[nodiscard]] std::vector<T> forward(std::span<const T> diagram, size_t batch_size,
                                         size_t n_pairs) const;

private:
    Config config_;
};

// Vectorization Layer (Diagram -> feature vector)

template <typename T = float>
class DiagramVectorizer
{
public:
    struct Config
    {
        enum class Method
        {
            PERSISTENCE_STATS, // Mean, std, max persistence
            BETTI_CURVE,       // Betti numbers over filtration
            ENTROPY,           // Persistence entropy
            LANDSCAPE          // Persistence landscape samples
        } method = Method::PERSISTENCE_STATS;

        int output_dim = 64;
        int n_bins = 10;
    };

    explicit DiagramVectorizer(Config config);

    // Vectorize diagram to fixed-size feature vector
    // Input: [batch, n_pairs, 3]
    // Output: [batch, output_dim]
    [[nodiscard]] std::vector<T> forward(std::span<const T> diagram, size_t batch_size,
                                         size_t n_pairs) const;

private:
    Config config_;

    [[nodiscard]] std::vector<T> persistence_stats(std::span<const T> diagram, size_t batch_size,
                                                   size_t n_pairs) const;

    [[nodiscard]] std::vector<T> betti_curve(std::span<const T> diagram, size_t batch_size,
                                             size_t n_pairs) const;
};

// GPU Kernels (CUDA)

#ifdef NERVE_USE_CUDA

template <typename T>
__global__ void
diagram_conv1d_cuda_kernel(const T *diagram,  // [batch, n_pairs, 3]
                           const T *features, // [batch, n_pairs, in_channels]
                           const T *kernel,   // [out_channels, in_channels * 2, kernel_size]
                           const T *bias,     // [out_channels]
                           T *output,         // [batch, out_channels, n_pairs]
                           int batch_size, int n_pairs, int in_channels, int out_channels,
                           int kernel_size);

template <typename T>
__global__ void persistence_image_cuda_kernel(const T *diagram, // [batch, n_pairs, 3]
                                              T *image,         // [batch, height, width]
                                              int batch_size, int n_pairs, int height, int width,
                                              T sigma, T min_birth, T max_death);

#endif // NERVE_USE_CUDA

} // namespace nerve::nn
