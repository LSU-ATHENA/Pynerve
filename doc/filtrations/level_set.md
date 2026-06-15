# Level set filtrations

Level set filtrations build a topological summary of a scalar function
$f: X \to \mathbb{R}$ defined on a point cloud, grid, or mesh. The
**sublevel set filtration** tracks how the topology of
$f^{-1}(-\infty, t]$ changes with $t$; the **superlevel set filtration**
tracks $f^{-1}[t, \infty)$.

This is the natural filtration for image analysis, scientific computing,
and any setting where data carries a scalar field (density, intensity,
distance to a reference, etc.).


## Definition

### Sublevel sets

Given a scalar function $f$ on a domain $X$, the **sublevel set** at
threshold $t$ is:

$$
X_t = \{x \in X : f(x) \leq t\}
$$

As $t$ increases from $-\infty$ to $\infty$, the sublevel sets grow:
$X_{t_1} \subseteq X_{t_2}$ for $t_1 \leq t_2$. This nested sequence is
the **sublevel set filtration**.

Persistent homology tracks:
- **Birth** of a component: a local minimum of $f$ is crossed.
- **Death** (merger) of components: a saddle point joins two minima basins.
- **Higher homology:** loops appear at saddle points and disappear at
  local maxima (in 2D) or at index-2 critical points (in 3D).

### Superlevel sets

The **superlevel set** at threshold $t$ is:

$$
X^t = \{x \in X : f(x) \geq t\}
$$

As $t$ decreases from $\infty$ to $-\infty$, superlevel sets grow.
The superlevel set filtration is the sublevel set filtration of $-f$,
with birth/death reversed: features born at high $f$ values die as $t$
drops.

### Relationship to Morse theory

For a Morse function $f$ on a compact manifold $M$, the sublevel set
filtration changes topology only at **critical points** of $f$. The
index of each critical point determines the homological change:

A minimum (index 0) gives birth to an $H_0$ component. A saddle in 2D (index 1) causes either an $H_0$ merger or an $H_1$ birth. A maximum in 2D or saddle in 3D (index 2) produces an $H_1$ death or $H_2$ birth. A maximum in 3D (index 3) causes an $H_2$ death.

### Cubical complex

For functions defined on a regular grid (pixels, voxels), the domain is
a cubical complex. Each cell (vertex, edge, square, cube) is assigned a
filtration value equal to the maximum (for sublevel) or minimum (for
superlevel) of $f$ over its vertices.

The **cubical complex** is preferred over VR for grid data because:
- It is smaller: $O(n^d)$ cells vs $O(n^{kd})$ simplices in VR.
- It respects the grid topology exactly (no spurious diagonals).
- It is faster to construct (no distance computations).


## Scalar field data

Level set filtrations apply to any setting where each point has an
associated scalar value:

### Density estimation

Given a probability density $p(x)$ estimated from samples:

- **Sublevel set** on $-p(x)$: finds modes (peaks) of the density.
- **Superlevel set** on $p(x)$: finds high-density regions (HDRs).

```python
import numpy as np
from sklearn.neighbors import KernelDensity

# Estimate density from point cloud
kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(points)
log_density = kde.score_samples(points)

# Sublevel set on negative density captures peaks
augmented = np.column_stack([points, -log_density])
result = pynerve.compute_persistence(augmented, max_dim=2, max_radius=2.0)
```

### Distance functions

For a reference set $S \subset X$, the distance function
$f(x) = d(x, S)$ produces:

- **Sublevel set:** thickening of $S$ (offset filtration).
- **Superlevel set:** points far from $S$.

This is equivalent to the VR complex of $S$ but computed on the
distance function.

### Intensity fields

In image analysis, the intensity $I(x, y)$ (or $I(x, y, z)$ for volumes)
is the scalar function:

- **Sublevel set:** dark regions appear first (valleys).
- **Superlevel set:** bright regions appear first (peaks).


## Image analysis examples

### Example 1: 2D image sublevel filtration

```python
import numpy as np
import pynerve

# Synthetic image with two bright blobs
x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
blob1 = np.exp(-((x - 1)**2 + y**2) / 0.5)
blob2 = np.exp(-((x + 1)**2 + y**2) / 0.5)
image = blob1 + blob2 + 0.1 * np.random.randn(100, 100)

# Convert grid to point cloud with intensity values
points = np.column_stack([x.ravel(), y.ravel(), image.ravel()])

# Sublevel set on -intensity captures bright blobs as H0 features
result = pynerve.compute_persistence(points, max_dim=2, max_radius=2.0)

# The two blobs appear as two persistent H0 births
pairs = result.pairs
h0_pairs = [(b, d) for (b, d, dim) in pairs if dim == 0]
print(f"Number of persistent components: {len([p for p in h0_pairs if p[1] - p[0] > 0.1])}")
```

### Example 2: MRI volume analysis

```python
import numpy as np
import pynerve

# Mock MRI volume (64x64x64)
volume = np.random.randn(64, 64, 64) * 0.1
volume[20:40, 20:40, 20:40] += 2.0  # bright region (tumor)

# Convert to point cloud with intensity
x, y, z = np.meshgrid(
    np.arange(64), np.arange(64), np.arange(64),
    indexing="ij",
)
points = np.column_stack([
    x.ravel(), y.ravel(), z.ravel(),
    volume.ravel(),
])

# Superlevel set captures bright regions as persistent features
result = pynerve.compute_persistence(points, max_dim=3, max_radius=5.0)
```

### Example 3: Topological denoising

```python
import numpy as np
import pynerve

# Noisy image with a single dark circle
x, y = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
circle_rim = ((x**2 + y**2) > 0.8) & ((x**2 + y**2) < 1.2)
image = np.where(circle_rim, -1.0, 0.0) + 0.2 * np.random.randn(200, 200)

# Sublevel filtration: the circle appears as a persistent H1 feature
points = np.column_stack([x.ravel(), y.ravel(), image.ravel()])
result = pynerve.compute_persistence(points, max_dim=2, max_radius=1.0)

# Filter out noisy H1 features by persistence threshold
pairs = result.pairs
significant_h1 = [(b, d) for (b, d, dim) in pairs
                  if dim == 1 and (d - b) > 0.3]
print(f"Significant loops: {len(significant_h1)}")
```

### Example 4: Connected component analysis

```python
import numpy as np
import pynerve

# Points with a density field: two clusters connected by a bridge
n_per_cluster = 500
cluster1 = np.random.randn(n_per_cluster, 2) + np.array([-2, 0])
cluster2 = np.random.randn(n_per_cluster, 2) + np.array([2, 0])
bridge = np.random.randn(50, 2) * 0.3  # sparse bridge
points = np.vstack([cluster1, cluster2, bridge])

# Compute density as scalar field
from sklearn.neighbors import KernelDensity
kde = KernelDensity(bandwidth=0.5).fit(points)
log_density = kde.score_samples(points)

# Sublevel set on intensity
augmented = np.column_stack([points, log_density])
result = pynerve.compute_persistence(augmented, max_dim=2, max_radius=2.0)

# Recovery of connected components:
# At low density thresholds, two separate components
# At higher threshold, the bridge merges them
# The merge point in the diagram indicates the bridge density
```


## API

### Level set persistence via VR on scalar-augmented coordinates

The primary workflow computes level set persistence by augmenting spatial
coordinates with the scalar function value, then applying VR persistence
in $\mathbb{R}^{d+1}$:

```python
import numpy as np
import pynerve

# Points with scalar values
#   points : (n, d) -- spatial coordinates
#   values : (n,)   -- scalar function at each point
points = np.random.randn(2000, 3)
values = np.linalg.norm(points, axis=1)  # e.g., distance from origin

# Sublevel set persistence via VR in R^(d+1)
augmented = np.column_stack([points, values])
result = pynerve.compute_persistence(augmented, max_dim=2, max_radius=2.0)

# The resulting diagram captures how sublevel sets of the
# scalar function evolve. Births correspond to critical values
# where new components appear or topology changes.
```

### Sublevel vs superlevel

```python
# Sublevel set (features at low scalar values)
augmented = np.column_stack([points, values])
result_sub = pynerve.compute_persistence(augmented, max_dim=2)

# Superlevel set (features at high scalar values)
augmented_super = np.column_stack([points, -values])
result_super = pynerve.compute_persistence(augmented_super, max_dim=2)
```

### PyTorch cubical (grid-based)

For functions defined on regular grids (images, volumes), use the cubical
complex approach:

```python
from pynerve.diff import DifferentiableCubical

cubical = DifferentiableCubical(max_dim=2, sublevel=True)

# Input: (batch, height, width) or (batch, depth, height, width)
# image = torch.randn(4, 64, 64)
# diagram = cubical(image)  # raises RuntimeError -- in preview
```

### C++ level set (core engine)

The level set engine is exposed through the C++ `nerve_internal` backend.
It supports structured grids (2D/3D) and mesh-based connectivity:

```cpp
// C++ API (via nerve_internal)
LevelSet ls;
ls.setGridShape(width, height);
ls.setFiltrationType("sublevel");  // or "superlevel"
ls.buildFiltration(scalar_field);  // builds sublevel/superlevel filtration
auto result = ls.computePersistence(max_dim);
```


## Recovery of connected components

Level set filtrations are particularly effective for recovering the
**connected components** of sublevel/superlevel sets and their hierarchy.

### Merge tree

The **merge tree** (also called the **contour tree**) records:
- When a new component appears (birth at a local minimum).
- When two components merge (death at a saddle point).

The persistence diagram of the $H_0$ sublevel set filtration is a
summary of the merge tree:

```python
import numpy as np
import pynerve

# 1D example: function with two valleys
x = np.linspace(0, 10, 1000)
f = np.sin(x) + 0.5 * np.sin(3 * x)

# Build point cloud in 2D (x, f(x))
points = np.column_stack([x, f])

# Sublevel set filtration
result = pynerve.compute_persistence(points, max_dim=1, max_radius=2.0)

# H0 pairs: (valley depth, merge height)
h0_pairs = [(b, d) for (b, d, dim) in result.pairs if dim == 0 and d > b]
for birth, death in h0_pairs:
    print(f"Component: birth at f={birth:.3f}, merges at f={death:.3f}")
```

### Hierarchical segmentation

In image analysis, the level set filtration provides a natural
**hierarchical segmentation**:

- Each local minimum (for sublevel) or maximum (for superlevel)
  seeds a region.
- As the threshold changes, regions merge at saddle points.
- The persistence diagram records the lifetime of each region.

### Watershed transform

The watershed transform segments an image into catchment basins of the
gradient. The **watershed hierarchy** is exactly the merge tree of the
sublevel set filtration of the gradient magnitude:

```python
# Pseudocode -- available via nerve_internal C++ backend
# ws = Watershed()
# ws.setInput(gradient_magnitude_image)
# ws.computeHierarchy()
# segments = ws.extractSegments(persistence_threshold=0.1)
```

Pynerve's `Watershed` class in the C++ backend builds watershed filtrations
from scalar fields, providing:
- Hierarchical segmentation with persistence-based simplification.
- Direct extraction of segments above a persistence threshold.


## When to use level set filtrations

For **image analysis**, apply level set filtration on pixel intensity. For **density estimation**, use level set on log-density values. For **distance functions**, compute level set on distance to a reference set. For **scientific computing**, apply level set on simulation output fields. For **mesh data**, use level set with mesh adjacency. For **Morse theory analysis**, combine sublevel or superlevel filtration with critical point extraction. For **topological simplification**, apply level set with persistence threshold filtering. For **time-varying data**, compute level set per timestep with tracking across frames.

## When not to use level set filtrations

When **no scalar field is available**, use VR or Alpha on coordinates alone instead. For **graph-structured data**, use path homology or persistent homology on the graph. For **manifold-valued data**, use VR with a manifold distance. When **only pairwise distances** are available, use VR or the witness complex.


## Related concepts

The level set construction is closely related to:

### Morse-Smale complex

Decomposes the domain into regions with uniform gradient flow behavior.
Pynerve provides `MorseSmale` in the C++ backend for computing gradient
flows, separatrices, and attractor basins:

```cpp
// C++ API (via nerve_internal)
ms = MorseSmale(ls)
ms.classifyCriticalPoints()
ms.computeSeparatrices()
```

### Watershed segmentation

The level set filtration provides a natural hierarchical segmentation.
Pynerve's `Watershed` class builds watershed filtrations from scalar fields:

```cpp
// C++ API (via nerve_internal)
ws = Watershed()
ws.setInput(scalar_field)
ws.computeHierarchy()
segments = ws.extractSegments(persistence_threshold=0.01)
```

### Discrete Morse theory

The `DiscreteMorse` engine computes Morse matchings on the filtration,
reducing complex size before reduction:

```cpp
// C++ API (via nerve_internal)
dm = DiscreteMorse(ls)
dm.computeMorseMatching()
dm.reduceComplex()
result = dm.computePersistence(max_dim)
```

### Persistent homology of the merge tree

The $H_0$ sublevel set persistence diagram is equivalent to the
**merge tree** (or **contour tree**) of the scalar function. This
connection enables:

- Extraction of the merge tree from the persistence diagram.
- Hierarchical segmentation with persistence-based simplification.
- Topological simplification: remove branches whose lifetime is below
  a threshold.


## Practical guidance

### Scale matters

The scalar values should be on a meaningful scale. Normalize to $[0, 1]$
or z-score if the function range is extreme:

```python
from scipy import stats

# Z-score normalization
values = stats.zscore(values)

# Or min-max normalization
values = (values - values.min()) / (values.max() - values.min())
```

### Grid resolution

For image data, consider downsampling before computing persistence --
topological features at the native resolution are dominated by noise:

```python
from skimage.transform import resize

# Downsample 1024x1024 to 256x256
image_low = resize(image, (256, 256), anti_aliasing=True)
```

### Sublevel vs. superlevel

With the **sublevel** filtration, low scalar values are born first and die later, making it suitable for analyzing valleys, minima, and dark regions. With the **superlevel** filtration, low values are born later and die first, making it suitable for analyzing peaks, maxima, and bright regions.

- Use **sublevel** for functions where low values represent features of
  interest (e.g., density valleys, distance to reference, dark areas).
- Use **superlevel** for functions where high values represent features
  (e.g., density peaks, bright areas, temperature hot spots).

### Connectivity

The level set complex requires a definition of adjacency (grid neighbors,
Delaunay edges, or kNN graph). For point clouds with a scalar function,
the VR complex on the augmented coordinates ($d+1$ dimensions) captures
the sublevel set topology.

For grid data, the cubical complex provides:
- **4-connectivity** (2D): each pixel connects to its N, S, E, W neighbors.
- **6-connectivity** (3D): each voxel connects to its 6 face-sharing neighbors.

### Preprocessing

```python
# Recommended preprocessing pipeline
import numpy as np
from scipy.ndimage import gaussian_filter

# 1. Smooth the scalar field to reduce noise
smoothed = gaussian_filter(scalar_field, sigma=2.0)

# 2. Normalize to [0, 1]
smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

# 3. Convert to point cloud + scalar
# (if not already on a grid)
```


## FAQ

### What is the difference between sublevel and superlevel filtrations?

Sublevel set filtrations track the evolution of the set of points where the scalar function is below a threshold: $f^{-1}(-\infty, t]$. As $t$ increases, the sublevel sets grow. Features born at low scalar values (valleys, minima, dark regions) appear first. Superlevel set filtrations track $f^{-1}[t, \infty)$, growing as $t$ decreases. They capture features at high scalar values (peaks, maxima, bright regions) first. The superlevel filtration of $f$ is equivalent to the sublevel filtration of $-f$, with the roles of birth and death reversed.

### When should I use cubical complexes vs VR for image data?

Cubical complexes are almost always preferred for regular grid data (images, volumes). They are significantly more compact, with $O(n^d)$ cells versus $O(n^{kd})$ simplices in a VR complex. Cubical complexes respect grid topology exactly without introducing spurious diagonal connections, and they avoid expensive distance computations. Use VR only when your data is an unstructured point cloud rather than a regular grid.

### How do I interpret persistence diagrams from scalar fields?

In a persistence diagram from a level set filtration, each point represents a topological feature (connected component, loop, void) tracked across thresholds. The birth coordinate indicates the scalar value where the feature first appears, and the death coordinate indicates where it disappears (merges into another feature or vanishes). Points far from the diagonal (long persistence) correspond to significant features, while points near the diagonal are likely noise. For sublevel filtrations, $H_0$ points represent local minima and their merger heights; $H_1$ points represent loops that appear and fill in.

### What preprocessing should I do on scalar fields?

Start by smoothing the scalar field with a Gaussian filter to suppress noise that creates spurious critical points. Normalize the scalar values, typically to $[0, 1]$ or via z-score, to ensure the persistence scale is interpretable. For large grids, consider downsampling to reduce computational cost -- topological features at native resolution are often dominated by noise. Finally, choose the appropriate filtration direction (sublevel for low-value features, superlevel for high-value features) based on what you want to detect.

### How does Morse theory relate to level set persistence?

Morse theory guarantees that the topology of sublevel sets changes only at critical points of the scalar function. For a Morse function on a compact manifold, each critical point has an index that determines the homological change: minima (index 0) create $H_0$ components, saddles merge or create higher homology, and maxima (index $d$) destroy top-dimensional homology. The persistence diagram records these critical events and pairs them: each minimum is paired with the saddle where its component merges, each saddle with the maximum where a loop is filled, and so on. The result is a multi-scale summary of the function's critical structure.
