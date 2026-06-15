
#pragma once

#include <vector>

namespace nerve::regularization
{
namespace gpu
{

struct RegularizerBenchmark
{
    double cpu_betti_ms;
    double gpu_betti_ms;
    double cpu_augment_ms;
    double gpu_augment_ms;
    double speedup_betti;
    double speedup_augment;
    int num_pairs;
    int feature_dim;
};

RegularizerBenchmark benchmarkRegularizer(int num_pairs, int feature_dim);

struct AugmentationBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    int num_samples;
    int feature_dim;
    int num_augmentations;
};

AugmentationBenchmark benchmarkAugmentation(int num_samples, int feature_dim, int num_augs);

struct LossBenchmark
{
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    int num_betti_dims;
    int num_persistence_pairs;
};

LossBenchmark benchmarkLoss(int num_dims, int num_pairs);

} // namespace gpu
} // namespace nerve::regularization
