#pragma once
#include "nerve/core_types.hpp"

#include <span>
#include <vector>

namespace nerve::algorithms
{

struct PersistenceLandscape
{
    std::vector<std::vector<double>> landscape_levels;
    double x_min, x_max;
    int num_levels;
};

struct PersistenceImage
{
    std::vector<std::vector<double>> image;
    int resolution;
    double sigma;
    double birth_min, birth_max, persistence_min, persistence_max;
};

struct PersistencePair
{
    double birth, death;
    int dimension;
};

template <typename T>
PersistenceLandscape compute_landscape(std::span<const T> diagram, size_t num_pairs,
                                       int num_levels = 5, double resolution = 0.01);

template <typename T>
PersistenceImage compute_persistence_image(std::span<const T> diagram, size_t num_pairs,
                                           int resolution = 64, double sigma = 0.1);

template <typename T>
std::vector<std::pair<double, int>> compute_betti_curve(std::span<const T> diagram,
                                                        size_t num_pairs, int max_dim = -1);

} // namespace nerve::algorithms
