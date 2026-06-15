bool canFormH1Feature(const Edge& edge,
                      const std::vector<Point>& points,
                      double max_radius,
                      const ReducedVRH1Config& config) {
    if (!config.use_triangle_filter) {
        return true;
    }
    const double radius = std::min(max_radius, config.max_radius);
    if (!std::isfinite(radius) || radius < 0.0) {
        return false;
    }
    const size_t v1 = static_cast<size_t>(edge.v1);
    const size_t v2 = static_cast<size_t>(edge.v2);
    const Point& p1 = points[edge.v1];
    const Point& p2 = points[edge.v2];
    for (size_t i = 0; i < points.size(); ++i) {
        if (i == v1 || i == v2) {
            continue;
        }
        const Point& p3 = points[i];
        const double d13 = p1.distance(p3);
        const double d23 = p2.distance(p3);
        if (d13 <= radius && d23 <= radius) {
            const double max_edge = std::max({edge.weight, d13, d23});
            if (max_edge <= radius) {
                return true;
            }
        }
    }
    return false;
}

std::vector<Edge> pruneByLocalConnectivity(
    const std::vector<Edge>& edges,
    const std::vector<Point>& points,
    const ReducedVRH1Config& config) {
    std::vector<Edge> pruned;
    std::vector<std::vector<int>> adjacency(points.size());
    for (const auto& e : edges) {
        adjacency[e.v1].push_back(e.v2);
        adjacency[e.v2].push_back(e.v1);
    }
    for (const auto& e : edges) {
        const int deg1 = static_cast<int>(adjacency[e.v1].size());
        const int deg2 = static_cast<int>(adjacency[e.v2].size());
        if (deg1 >= 2 && deg2 >= 2) {
            pruned.push_back(e);
        } else if (config.preserve_connectivity) {
            std::vector<int> parent(points.size());
            std::iota(parent.begin(), parent.end(), 0);
            auto find = [&](int x, auto&& find_ref) -> int {
                if (parent[x] != x) {
                    parent[x] = find_ref(parent[x], find_ref);
                }
                return parent[x];
            };
            for (const auto& other_e : edges) {
                if (other_e.v1 == e.v1 && other_e.v2 == e.v2) {
                    continue;
                }
                const int root1 = find(other_e.v1, find);
                const int root2 = find(other_e.v2, find);
                if (root1 != root2) {
                    parent[root1] = root2;
                }
            }
            const int root1 = find(e.v1, find);
            const int root2 = find(e.v2, find);
            if (root1 != root2) {
                pruned.push_back(e);
            }
        }
    }
    return pruned;
}

std::vector<Triangle> filterTriangles(
    const std::vector<Triangle>& triangles,
    const std::vector<Edge>& edges,
    const ReducedVRH1Config& config) {
    if (!config.use_triangle_filter) {
        return triangles;
    }
    std::vector<Triangle> filtered;
    auto edgeKey = [](int a, int b) -> uint64_t {
        if (a > b) {
            std::swap(a, b);
        }
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };
    std::unordered_set<uint64_t> edge_set;
    for (const auto& e : edges) {
        edge_set.insert(edgeKey(e.v1, e.v2));
    }
    for (const auto& t : triangles) {
        const bool e12_in = edge_set.count(edgeKey(t.v1, t.v2));
        const bool e13_in = edge_set.count(edgeKey(t.v1, t.v3));
        const bool e23_in = edge_set.count(edgeKey(t.v2, t.v3));
        if (e12_in && e13_in && e23_in) {
            filtered.push_back(t);
        }
    }
    return filtered;
}
