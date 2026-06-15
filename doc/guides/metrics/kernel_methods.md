# Kernel methods

Kernel functions can approximate distances as `d(x,y) = sqrt(K(x,x) + K(y,y) - 2*K(x,y))`:

```python
# Gaussian kernel -> distance
def gaussian_kernel(sigma=1.0):
    def kernel(a, b):
        diff = a - b
        return np.exp(-np.dot(diff, diff) / (2 * sigma ** 2))

    # Convert kernel to distance via feature map approximation
    from pynerve.algorithms import kernel_distance
    return kernel_distance(kernel)

result = pynerve.compute_persistence(
    points, max_dim=2, metric=gaussian_kernel(0.5)
)
```

### Kernel distance conversion

For a positive definite kernel K, the induced distance is:

```
d_K(x, y) = sqrt(K(x, x) + K(y, y) - 2*K(x, y))
```

Pynerve applies this conversion internally when a kernel function is provided. The kernel is evaluated O(n^2) times during VR construction.

### Kernel methods detail

Pynerve supports several kernel methods. The Gaussian kernel (`exp(-||x-y||^2 / 2*sigma^2)`) is used for universal approximation. The Linear kernel (`x * y`) provides cosine-like similarity. The Polynomial kernel (`(gamma * x * y + coef0)^d`) captures higher-order interactions. The Laplacian kernel (`exp(-||x-y|| / sigma)`) is robust to outliers. The Sigmoid kernel (`tanh(gamma * x * y + coef0)`) offers neural network similarity. The Chi-squared kernel (`exp(-sum((x_i - y_i)^2 / (x_i + y_i)))`) is well-suited to histogram data. The Hellinger kernel (`sqrt(sum(sqrt(x_i) - sqrt(y_i))^2)`) works well with probability distributions.

### Gaussian kernel specifics

```python
# Gaussian kernel with automatic sigma selection
from pynerve.algorithms import kernel_distance, estimate_sigma

# Heuristic: sigma = median pairwise distance / 2
sigma = estimate_sigma(points)

def gaussian(x, y):
    diff = x - y
    return np.exp(-np.dot(diff, diff) / (2 * sigma * sigma))

result = pynerve.compute_persistence(
    points, max_dim=2, metric=kernel_distance(gaussian)
)
```

### Polynomial kernel

```python
# Polynomial kernel of degree 3
def polynomial_kernel(degree=3, gamma=1.0, coef0=1.0):
    def kernel(a, b):
        return (gamma * np.dot(a, b) + coef0) ** degree

    from pynerve.algorithms import kernel_distance
    return kernel_distance(kernel)

result = pynerve.compute_persistence(
    points, max_dim=2, metric=polynomial_kernel(3)
)
```

The polynomial kernel induces a distance that captures higher-order interactions between features. Increasing the degree increases the expressiveness at the cost of numerical range.

[Back to index](index.md)
