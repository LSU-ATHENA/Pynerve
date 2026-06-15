#include "nerve/algorithms/mapper.hpp"
#include "nerve/algorithms/persistence_vectorization.hpp"
#include "nerve/nn/diagram_conv.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

template <typename T>
bool approxEqual(T a, T b, T eps = T(1e-6))
{
    return std::abs(a - b) <= eps * std::max(T(1), std::max(std::abs(a), std::abs(b)));
}

std::vector<double> buildDiagonal2D(size_t n)
{
    std::vector<double> pts(n * 2);
    for (size_t i = 0; i < n; ++i)
    {
        pts[i * 2 + 0] = static_cast<double>(i);
        pts[i * 2 + 1] = static_cast<double>(i);
    }
    return pts;
}

std::vector<double> buildGaussianCluster3D(size_t n, double cx, double cy, double cz, double noise)
{
    std::vector<double> pts(n * 3);
    for (size_t i = 0; i < n; ++i)
    {
        double angle = 6.283185307179586 * static_cast<double>(i) / static_cast<double>(n);
        pts[i * 3 + 0] = cx + std::cos(angle) * 2.0 + (static_cast<double>(i % 7) - 3.0) * noise;
        pts[i * 3 + 1] =
            cy + std::sin(angle) * 2.0 + (static_cast<double>((i * 3) % 11) - 5.0) * noise;
        pts[i * 3 + 2] = cz + (static_cast<double>(i) / static_cast<double>(n)) * 4.0 +
                         (static_cast<double>((i * 5) % 9) - 4.0) * noise;
    }
    return pts;
}

} // namespace

int main()
{
    // PCA filter tests
    {
        nerve::algorithms::PCAFilter<double> pca_2d(2);
        auto pts = buildDiagonal2D(10);
        auto filter_vals = pca_2d.apply(pts, 10, 2);
        assert(!filter_vals.empty());
        assert(filter_vals.size() == 20);
        for (double v : filter_vals)
        {
            assert(std::isfinite(v));
        }
    }

    {
        nerve::algorithms::PCAFilter<double> pca_3d(3);
        auto pts_3d = buildGaussianCluster3D(20, 0.0, 0.0, 0.0, 0.05);
        auto filter_vals = pca_3d.apply(pts_3d, 20, 3);
        assert(!filter_vals.empty());
        assert(filter_vals.size() == 60);
        for (double v : filter_vals)
        {
            assert(std::isfinite(v));
        }
    }

    {
        nerve::algorithms::PCAFilter<double> pca_2comp(2);
        std::vector<double> cloud_100d(50 * 100);
        for (size_t i = 0; i < 50; ++i)
        {
            for (size_t d = 0; d < 100; ++d)
            {
                cloud_100d[i * 100 + d] = static_cast<double>(i + d) * 0.01;
            }
        }
        auto filter_vals = pca_2comp.apply(cloud_100d, 50, 100);
        assert(!filter_vals.empty());
        assert(filter_vals.size() == 100);
        for (double v : filter_vals)
        {
            assert(std::isfinite(v));
        }
    }

    {
        nerve::algorithms::PCAFilter<double> pca_empty(2);
        auto empty_vals = pca_empty.apply({}, 0, 2);
        assert(empty_vals.empty());
    }

    // Density filter tests
    {
        nerve::algorithms::DensityFilter<double> density(5);
        auto pts = buildGaussianCluster3D(10, 0.0, 0.0, 0.0, 0.01);
        auto density_vals = density.apply(pts, 10, 3);
        assert(!density_vals.empty());
        assert(density_vals.size() == 10);
        for (double v : density_vals)
        {
            assert(std::isfinite(v));
            assert(v >= 0.0);
        }
        assert(density.name() == "density");
    }

    {
        nerve::algorithms::DensityFilter<double> density_empty(3);
        auto empty_vals = density_empty.apply({}, 0, 3);
        assert(empty_vals.empty());
    }

    // Eccentricity filter tests
    {
        nerve::algorithms::EccentricityFilter<double> ecc;
        auto pts = buildGaussianCluster3D(10, 0.0, 0.0, 0.0, 0.01);
        auto ecc_vals = ecc.apply(pts, 10, 3);
        assert(!ecc_vals.empty());
        assert(ecc_vals.size() == 10);
        for (double v : ecc_vals)
        {
            assert(std::isfinite(v));
            assert(v >= 0.0);
        }
        assert(ecc.name() == "eccentricity");
    }

    {
        nerve::algorithms::EccentricityFilter<double> ecc;
        auto pts = buildDiagonal2D(4);
        auto ecc_vals = ecc.apply(pts, 4, 2);
        assert(ecc_vals.size() == 4);
        double max_ecc = ecc_vals[0];
        for (double v : ecc_vals)
        {
            if (v > max_ecc)
                max_ecc = v;
        }
        assert(max_ecc > 0.0);
    }

    // Full Mapper pipeline
    {
        auto pts = buildGaussianCluster3D(30, 0.0, 0.0, 0.0, 0.02);
        auto filter = std::make_shared<nerve::algorithms::PCAFilter<double>>(1);
        auto clusterer = std::make_shared<nerve::algorithms::DBSCANClustering<double>>(
            nerve::algorithms::DBSCANClustering<double>::Config{0.3, 3});

        nerve::algorithms::MapperAlgorithm<double> mapper(
            {filter, 8, 0.25, clusterer, true, false});
        auto result = mapper.compute(pts, 30, 3);
        assert(!result.filter_values.empty());
        assert(result.filter_values.size() == 30);
    }

    // Persistence image tests via PersistenceImageLayer
    {
        nerve::nn::PersistenceImageLayer<float>::Config config;
        config.resolution_h = 20;
        config.resolution_w = 20;
        config.sigma = 0.1f;
        config.weight = nerve::nn::PersistenceImageLayer<float>::Config::Weight::LINEAR;

        nerve::nn::PersistenceImageLayer<float> layer(config);

        std::vector<float> diagram = {0.1f, 0.5f, 0.0f, 0.2f, 0.8f, 0.0f, 0.3f, 0.4f, 0.0f};
        size_t n_pairs = 3;
        size_t batch_size = 1;

        auto image = layer.forward(diagram, batch_size, n_pairs);
        assert(!image.empty());
        assert(image.size() == static_cast<size_t>(20 * 20));
        float sum = 0.0f;
        for (float v : image)
        {
            assert(std::isfinite(v));
            sum += v;
        }
        assert(sum >= 0.0f);
    }

    {
        nerve::nn::PersistenceImageLayer<float>::Config config;
        config.resolution_h = 10;
        config.resolution_w = 10;
        config.sigma = 0.1f;
        config.weight = nerve::nn::PersistenceImageLayer<float>::Config::Weight::QUADRATIC;
        nerve::nn::PersistenceImageLayer<float> layer(config);

        std::vector<float> empty_diagram;
        auto image = layer.forward(empty_diagram, 1, 0);
        assert(!image.empty());
        assert(image.size() == 100);
    }

    // Persistence landscape tests via LandscapeLayer
    {
        nerve::nn::LandscapeLayer<float>::Config config;
        config.n_layers = 3;
        config.resolution = 50;
        config.min_persistence = 0.0f;

        nerve::nn::LandscapeLayer<float> layer(config);

        std::vector<float> diagram = {0.1f, 0.5f, 0.0f, 0.2f, 0.8f, 0.0f};
        size_t n_pairs = 2;
        size_t batch_size = 1;

        auto landscape = layer.forward(diagram, batch_size, n_pairs);
        assert(!landscape.empty());
        assert(landscape.size() == static_cast<size_t>(3 * 50));
        for (float v : landscape)
        {
            assert(std::isfinite(v));
        }
    }

    // Betti curve test via DiagramVectorizer
    {
        nerve::nn::DiagramVectorizer<float>::Config config;
        config.method = nerve::nn::DiagramVectorizer<float>::Config::Method::BETTI_CURVE;
        config.output_dim = 64;
        config.n_bins = 10;

        nerve::nn::DiagramVectorizer<float> vectorizer(config);

        std::vector<float> diagram = {0.0f, 0.3f, 0.0f, 0.0f, 2.0f, 0.0f, 0.5f, 0.8f, 1.0f};
        size_t n_pairs = 3;
        size_t batch_size = 1;

        auto vec = vectorizer.forward(diagram, batch_size, n_pairs);
        assert(!vec.empty());
        assert(vec.size() == 64);
        for (float v : vec)
        {
            assert(std::isfinite(v));
        }
    }

    // Persistence stats method
    {
        nerve::nn::DiagramVectorizer<float>::Config config;
        config.method = nerve::nn::DiagramVectorizer<float>::Config::Method::PERSISTENCE_STATS;
        config.output_dim = 32;

        nerve::nn::DiagramVectorizer<float> vectorizer(config);

        std::vector<float> diagram = {0.1f, 0.5f, 0.0f, 0.2f, 0.8f, 0.0f};
        size_t n_pairs = 2;
        size_t batch_size = 1;

        auto vec = vectorizer.forward(diagram, batch_size, n_pairs);
        assert(!vec.empty());
        assert(vec.size() == 32);
    }

    // Nerve namespace vectorization API
    {
        std::vector<double> diagram = {0.1, 0.5, 0.0, 0.2, 0.8, 0.0, 0.3, 0.4, 0.1};
        size_t n_pairs = 3;

        auto landscape = nerve::algorithms::compute_landscape<double>(diagram, n_pairs, 3, 0.05);
        assert(landscape.num_levels == 3);
        assert(!landscape.landscape_levels.empty());
        assert(landscape.landscape_levels[0].size() > 0);
        for (double v : landscape.landscape_levels[0])
        {
            assert(std::isfinite(v));
        }

        auto image =
            nerve::algorithms::compute_persistence_image<double>(diagram, n_pairs, 32, 0.1);
        assert(image.resolution == 32);
        assert(!image.image.empty());
        assert(image.image[0].size() == 32);

        auto betti = nerve::algorithms::compute_betti_curve<double>(diagram, n_pairs);
        assert(!betti.empty());
    }

    // Custom filter
    {
        nerve::algorithms::CustomFilter<double> custom(
            [](std::span<const double> pts, size_t n, size_t dim) -> std::vector<double> {
                std::vector<double> result(n);
                for (size_t i = 0; i < n; ++i)
                {
                    double sum = 0.0;
                    for (size_t d = 0; d < dim; ++d)
                    {
                        sum += pts[i * dim + d];
                    }
                    result[i] = sum;
                }
                return result;
            },
            "sum");

        auto pts = buildDiagonal2D(5);
        auto vals = custom.apply(pts, 5, 2);
        assert(vals.size() == 5);
        assert(custom.name() == "sum");
        for (size_t i = 0; i < 5; ++i)
        {
            assert(approxEqual(vals[i], 2.0 * static_cast<double>(i)));
        }
    }

    return 0;
}
