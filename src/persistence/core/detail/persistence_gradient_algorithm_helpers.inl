namespace {

constinit const double GRADIENT_EPSILON = 1e-10;

double pointDistance(const MatrixXd& points, int i, int j) {
    double dist = 0.0;
    for (int d = 0; d < points.cols(); ++d) {
        const double diff = points(i, d) - points(j, d);
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

class UnionFind {
   public:
    explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int x) {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    void unite(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return;
        if (rank_[x] < rank_[y]) std::swap(x, y);
        parent_[y] = x;
        if (rank_[x] == rank_[y]) ++rank_[x];
    }

    bool connected(int x, int y) {
        return find(x) == find(y);
    }

   private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

struct Edge {
    int u;
    int v;
    double weight;

    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

}  // namespace
