#include "nerve/persistence/core/ph_gradient_basic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nerve::persistence::gradient
{

TopologyOptimizer::TopologyOptimizer(const Config &config)
    : config_(config)
{}

MatrixXd TopologyOptimizer::optimize(
    MatrixXd points, const DifferentiableDiagram &target_diagram,
    const std::function<DifferentiableDiagram(const MatrixXd &)> &ph_function)
{
    if (points.rows() == 0 || points.cols() == 0)
    {
        return points;
    }

    MatrixXd velocity(points.rows(), points.cols());
    double prev_loss = std::numeric_limits<double>::max();

    for (int iter = 0; iter < config_.max_iterations; ++iter)
    {
        const DifferentiableDiagram current_diagram = ph_function(points);

        const double loss = [&]() {
            double total = 0.0;
            const Size count = std::min(current_diagram.size(), target_diagram.size());
            for (Size i = 0; i < count; ++i)
            {
                const double db = current_diagram.persistence_pairs[i].first -
                                  target_diagram.persistence_pairs[i].first;
                const double dd = current_diagram.persistence_pairs[i].second -
                                  target_diagram.persistence_pairs[i].second;
                total += db * db + dd * dd;
            }
            if (count < target_diagram.size())
            {
                for (Size i = count; i < target_diagram.size(); ++i)
                {
                    const double b = target_diagram.persistence_pairs[i].first;
                    const double d = target_diagram.persistence_pairs[i].second;
                    total += b * b + d * d;
                }
            }
            return total;
        }();

        const double loss_change = std::abs(prev_loss - loss);
        if (loss_change < config_.convergence_threshold)
        {
            break;
        }
        prev_loss = loss;

        MatrixXd gradient(points.rows(), points.cols());
        for (int r = 0; r < points.rows(); ++r)
        {
            for (int c = 0; c < points.cols(); ++c)
            {
                const double original = points(r, c);
                const double h = std::max(1e-8, std::abs(original) * 1e-4);

                points(r, c) = original + h;
                const DifferentiableDiagram plus_diagram = ph_function(points);

                points(r, c) = original - h;
                const DifferentiableDiagram minus_diagram = ph_function(points);

                points(r, c) = original;

                const Size count =
                    std::min({plus_diagram.size(), minus_diagram.size(), target_diagram.size()});
                double plus_loss = 0.0;
                double minus_loss = 0.0;
                for (Size i = 0; i < count; ++i)
                {
                    {
                        const double db = plus_diagram.persistence_pairs[i].first -
                                          target_diagram.persistence_pairs[i].first;
                        const double dd = plus_diagram.persistence_pairs[i].second -
                                          target_diagram.persistence_pairs[i].second;
                        plus_loss += db * db + dd * dd;
                    }
                    {
                        const double db = minus_diagram.persistence_pairs[i].first -
                                          target_diagram.persistence_pairs[i].first;
                        const double dd = minus_diagram.persistence_pairs[i].second -
                                          target_diagram.persistence_pairs[i].second;
                        minus_loss += db * db + dd * dd;
                    }
                }
                if (count < target_diagram.size())
                {
                    for (Size i = count; i < target_diagram.size(); ++i)
                    {
                        const double b = target_diagram.persistence_pairs[i].first;
                        const double d = target_diagram.persistence_pairs[i].second;
                        const double extra = b * b + d * d;
                        plus_loss += extra;
                        minus_loss += extra;
                    }
                }

                gradient(r, c) = (plus_loss - minus_loss) / (2.0 * h);
            }
        }

        const double grad_norm = gradient.norm();
        if (grad_norm < 1e-15)
        {
            break;
        }

        const double lr = config_.learning_rate / std::max(1.0, grad_norm);

        for (int r = 0; r < points.rows(); ++r)
        {
            for (int c = 0; c < points.cols(); ++c)
            {
                if (config_.use_momentum)
                {
                    velocity(r, c) = config_.momentum_beta * velocity(r, c) - lr * gradient(r, c);
                    points(r, c) += velocity(r, c);
                }
                else
                {
                    points(r, c) -= lr * gradient(r, c);
                }
            }
        }
    }

    return points;
}

} // namespace nerve::persistence::gradient
