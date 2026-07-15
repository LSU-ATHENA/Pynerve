namespace
{

size_t validatePointData(std::span<const double> points_data, size_t n_points, size_t point_dim,
                         const char *operation)
{
    if (n_points > std::numeric_limits<size_t>::max() / point_dim)
    {
        throw std::length_error(std::string(operation) + ": point shape overflows size_t");
    }
    if (n_points > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        point_dim > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::length_error(std::string(operation) + ": point shape exceeds matrix limits");
    }

    const size_t required_values = n_points * point_dim;
    if (required_values > std::vector<double>().max_size())
    {
        throw std::length_error(std::string(operation) + ": point data exceeds vector capacity");
    }
    if (points_data.size() < required_values)
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": points_data is smaller than n_points * point_dim");
    }
    for (size_t i = 0; i < required_values; ++i)
    {
        if (!std::isfinite(points_data[i]))
        {
            throw std::invalid_argument(std::string(operation) +
                                        ": point coordinates must be finite");
        }
    }
    return required_values;
}

bool validPersistencePair(const std::pair<double, double> &pair)
{
    const bool finite_death = std::isfinite(pair.second);
    const bool infinite_death = pair.second == std::numeric_limits<double>::infinity();
    return std::isfinite(pair.first) && (finite_death || infinite_death) &&
           (!finite_death || pair.second >= pair.first);
}

void validateDifferentiableDiagram(const DifferentiableDiagram &diagram, const char *operation,
                                   bool require_dimensions)
{
    if (require_dimensions && diagram.dimensions.size() < diagram.persistence_pairs.size())
    {
        throw std::invalid_argument(std::string(operation) +
                                    ": diagram dimensions do not match persistence pairs");
    }
    for (size_t i = 0; i < diagram.persistence_pairs.size(); ++i)
    {
        if (!validPersistencePair(diagram.persistence_pairs[i]))
        {
            throw std::invalid_argument(std::string(operation) +
                                        ": diagram contains invalid persistence pairs");
        }
        if (i < diagram.dimensions.size() && diagram.dimensions[i] < 0)
        {
            throw std::invalid_argument(std::string(operation) +
                                        ": diagram contains negative dimensions");
        }
    }
}

} // namespace

StochasticPersistenceGradient::StochasticPersistenceGradient(int batch_size, double)
    : batch_size_(batch_size)
    , rng_(42)
{}

MatrixXd StochasticPersistenceGradient::computeStochasticGradient(
    const MatrixXd &points,
    const std::function<DifferentiableDiagram(const std::vector<int> &indices)> &ph_function,
    const std::vector<std::pair<double, double>> &loss_gradients)
{
    int n_points = points.rows();

    // Sample mini-batch
    auto batch_indices = sampleBatch(n_points);

    // Build mini-batch points
    MatrixXd batch_points(static_cast<int>(batch_indices.size()), points.cols());
    for (size_t i = 0; i < batch_indices.size(); ++i)
    {
        for (int d = 0; d < points.cols(); ++d)
        {
            batch_points(static_cast<int>(i), d) = points(batch_indices[i], d);
        }
    }

    // Compute PH on mini-batch
    DifferentiableDiagram batch_diagram = ph_function(batch_indices);

    // Compute gradient on mini-batch
    PersistenceGradient grad_computer;
    MatrixXd batch_grad =
        grad_computer.computeGradient(batch_points, batch_diagram, loss_gradients);

    // Map back to full gradient (sparse update)
    MatrixXd full_grad = MatrixXd::Zero(n_points, points.cols());
    for (size_t i = 0; i < batch_indices.size(); ++i)
    {
        for (int d = 0; d < points.cols(); ++d)
        {
            full_grad(batch_indices[i], d) = batch_grad(static_cast<int>(i), d);
        }
    }

    return full_grad;
}

std::vector<int> StochasticPersistenceGradient::sampleBatch(int n_points)
{
    std::vector<int> all_indices(n_points);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    std::shuffle(all_indices.begin(), all_indices.end(), rng_);

    int actual_batch = std::min(batch_size_, n_points);
    return std::vector<int>(all_indices.begin(), all_indices.begin() + actual_batch);
}

// TopologyOptimizer Implementation

TopologyOptimizer::TopologyOptimizer(const Config &config)
    : config_(config)
{}

MatrixXd TopologyOptimizer::optimize(
    MatrixXd points, const DifferentiableDiagram &target_diagram,
    const std::function<DifferentiableDiagram(const MatrixXd &)> &ph_function)
{
    MatrixXd velocity(points.rows(), points.cols());
    // Initialize velocity to zero
    for (int i = 0; i < velocity.rows(); ++i)
    {
        for (int j = 0; j < velocity.cols(); ++j)
        {
            velocity(i, j) = 0.0;
        }
    }

    PersistenceGradient grad_computer;

    for (int iter = 0; iter < config_.max_iterations; ++iter)
    {
        // Compute current diagram
        DifferentiableDiagram current_diagram = ph_function(points);

        // Compute topology optimization gradient
        MatrixXd grad = grad_computer.computeTopologyOptimizationGradient(points, current_diagram,
                                                                          target_diagram);

        // Momentum update
        if (config_.use_momentum)
        {
            velocity = velocity * config_.momentum_beta + grad * (1.0 - config_.momentum_beta);
            for (int i = 0; i < points.rows(); ++i)
            {
                for (int j = 0; j < points.cols(); ++j)
                {
                    points(i, j) -= config_.learning_rate * velocity(i, j);
                }
            }
        }
        else
        {
            for (int i = 0; i < points.rows(); ++i)
            {
                for (int j = 0; j < points.cols(); ++j)
                {
                    points(i, j) -= config_.learning_rate * grad(i, j);
                }
            }
        }

        // Check convergence
        if (grad.norm() < config_.convergence_threshold)
        {
            break;
        }
    }

    return points;
}

// High-level API Implementation

DifferentiableDiagram computeDifferentiable(std::span<const double> points_data, size_t n_points,
                                            size_t point_dim, double max_distance, int max_dim)
{
    DifferentiableDiagram result;
    if (point_dim == 0 || n_points == 0)
    {
        return result;
    }
    if (!std::isfinite(max_distance) || max_distance <= 0.0)
    {
        throw std::invalid_argument(
            "computeDifferentiable: max_distance must be finite and non-negative");
    }
    (void)validatePointData(points_data, n_points, point_dim, "computeDifferentiable");

    // Convert to MatrixXd
    MatrixXd points(static_cast<int>(n_points), static_cast<int>(point_dim));
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t d = 0; d < point_dim; ++d)
        {
            points(static_cast<int>(i), static_cast<int>(d)) = points_data[i * point_dim + d];
        }
    }

    // Build edge list for Vietoris-Rips complex
    std::vector<Edge> edges;
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t j = i + 1; j < n_points; ++j)
        {
            double dist = 0.0;
            for (size_t d = 0; d < point_dim; ++d)
            {
                double diff = points_data[i * point_dim + d] - points_data[j * point_dim + d];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            if (!std::isfinite(dist))
            {
                throw std::overflow_error(
                    "computeDifferentiable: distance computation produced a non-finite value");
            }
            if (dist <= max_distance)
            {
                edges.push_back({static_cast<int>(i), static_cast<int>(j), dist});
            }
        }
    }

    std::sort(edges.begin(), edges.end());

    // Compute 0D persistence using Union-Find
    UnionFind uf(static_cast<int>(n_points));

    for (const auto &edge : edges)
    {
        if (!uf.connected(edge.u, edge.v))
        {
            // This edge creates a 0D death
            result.persistence_pairs.push_back({0.0, edge.weight});
            result.dimensions.push_back(0);

            VectorXi birth_simplex(1);
            birth_simplex[0] = edge.u;
            result.birth_simplices.push_back(birth_simplex);

            VectorXi death_simplex(1);
            death_simplex[0] = edge.v;
            result.death_simplices.push_back(death_simplex);

            uf.unite(edge.u, edge.v);
        }
        else
        {
            // This edge creates a 1D birth (cycle)
            if (max_dim >= 1)
            {
                result.persistence_pairs.push_back(
                    {edge.weight, std::numeric_limits<double>::infinity()});
                result.dimensions.push_back(1);

                VectorXi birth_simplex(2);
                birth_simplex[0] = edge.u;
                birth_simplex[1] = edge.v;
                result.birth_simplices.push_back(birth_simplex);
                result.death_simplices.push_back(VectorXi());
            }
        }
    }

    // Add essential 0D features (connected components)
    std::unordered_map<int, int> components;
    for (size_t i = 0; i < n_points; ++i)
    {
        components[uf.find(static_cast<int>(i))]++;
    }

    for (const auto &component : components)
    {
        const int root = component.first;
        result.persistence_pairs.push_back({0.0, std::numeric_limits<double>::infinity()});
        result.dimensions.push_back(0);

        VectorXi birth_simplex(1);
        birth_simplex[0] = root;
        result.birth_simplices.push_back(birth_simplex);
        result.death_simplices.push_back(VectorXi());
    }

    return result;
}

std::vector<double> persistenceBackward(std::span<const double> points_data, size_t n_points,
                                        size_t point_dim, const DifferentiableDiagram &diagram,
                                        std::span<const double> grad_birth,
                                        std::span<const double> grad_death)
{
    if (point_dim == 0 || n_points == 0)
    {
        return {};
    }
    const size_t required_values =
        validatePointData(points_data, n_points, point_dim, "persistenceBackward");
    validateDifferentiableDiagram(diagram, "persistenceBackward", false);

    // Convert to MatrixXd
    MatrixXd points(static_cast<int>(n_points), static_cast<int>(point_dim));
    for (size_t i = 0; i < n_points; ++i)
    {
        for (size_t d = 0; d < point_dim; ++d)
        {
            points(static_cast<int>(i), static_cast<int>(d)) = points_data[i * point_dim + d];
        }
    }

    // Build loss gradients
    std::vector<std::pair<double, double>> loss_gradients;
    for (size_t i = 0; i < diagram.persistence_pairs.size(); ++i)
    {
        double gb = (i < grad_birth.size()) ? grad_birth[i] : 0.0;
        double gd = (i < grad_death.size()) ? grad_death[i] : 0.0;
        if (!std::isfinite(gb) || !std::isfinite(gd))
        {
            throw std::invalid_argument("persistenceBackward: gradients must be finite");
        }
        loss_gradients.push_back({gb, gd});
    }

    // Compute gradient
    PersistenceGradient computer;
    MatrixXd grad = computer.computeGradient(points, diagram, loss_gradients);

    // Convert back to vector
    std::vector<double> result;
    result.reserve(required_values);
    for (int i = 0; i < grad.rows(); ++i)
    {
        for (int j = 0; j < grad.cols(); ++j)
        {
            result.push_back(grad(i, j));
        }
    }

    return result;
}
