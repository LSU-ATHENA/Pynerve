# Examples: end-to-end

### Cup product analysis of 3D point clouds

```python
from pynerve.specialized import cup_product
import numpy as np

# Generate a 3D point cloud sampled from a torus
theta = np.random.rand(500) * 2 * np.pi
phi = np.random.rand(500) * 2 * np.pi
R, r = 2.0, 1.0
points = np.column_stack([
    (R + r * np.cos(theta)) * np.cos(phi),
    (R + r * np.cos(theta)) * np.sin(phi),
    r * np.sin(theta),
])

# Compute cup product on H^1
result = cup_product(points, max_dim=2, max_radius=3.0,
                     dims=[(1, 1)])

# Torus has H^1 = R^2 with non-trivial cup product
# beta_1 = 2, cup product of generators = generator of H^2
cup_table = result.cup_table
print(f"Cup product H^1 x H^1 -> H^2:")
print(cup_table[0])  # alpha U alpha = 0
print(cup_table[1])  # beta U beta = 0
print(cup_table[2])  # alpha U beta = gamma (non-zero)
```

### Reeb graph of terrain data

```python
from pynerve.specialized import reeb_graph, contour_tree
import numpy as np

# Generate terrain height field on a grid
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y) + 0.3 * np.random.randn(50, 50)

# Build graph from grid adjacency
vertices = np.column_stack([X.ravel(), Y.ravel()])
values = Z.ravel()

# Reeb graph of terrain
result = reeb_graph(
    adjacency=grid_adjacency(50, 50),
    function_values=values,
    persistence_threshold=0.1,
)

print(f"Nodes: {len(result.nodes)}")
print(f"Arcs: {len(result.arcs)}")

# Merge tree highlights mountain peaks and valleys
mt = result.merge_tree
peaks = mt.get_maxima()  # local maxima (peaks)
valleys = mt.get_minima()  # local minima (valleys)
```

### Zigzag persistence on time-varying data

```python
from pynerve.specialized import zigzag_persistence, ZigzagMatcher
import numpy as np

# Generate time-varying point cloud
n_points = 100
n_times = 20

time_slices = []
for t in range(n_times):
    # Points oscillate between two clusters
    angle = t / n_times * 2 * np.pi
    cluster1 = np.random.randn(n_points // 2, 3) + [np.cos(angle), 0, 0]
    cluster2 = np.random.randn(n_points // 2, 3) + [np.sin(angle), 0, 0]
    time_slices.append(np.vstack([cluster1, cluster2]))

# Compute zigzag persistence
result = zigzag_persistence(
    time_slices,
    max_dim=1,
    max_radius=2.0,
    backend="cuda",
)

print(f"Intervals: {len(result['pairs'])}")

# Match intervals across time to track feature evolution
matcher = ZigzagMatcher(distance_threshold=0.1)
matches = matcher.match_intervals(result)

for i, (interval, time_range) in enumerate(matches[:5]):
    print(f"Feature {i}: birth={interval.birth:.2f}, "
          f"active t={time_range.start}-{time_range.end}, "
          f"dim={interval.dimension}")
```


[Back to index](index.md)
