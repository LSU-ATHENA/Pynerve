# Example: summary comparison

```python
from pynerve import summarize
from pynerve.summary import compare_summaries

# Generate summaries for two point clouds
s1 = summarize(points_cloud_a, max_size=1024)
s2 = summarize(points_cloud_b, max_size=1024)

# Direct binary comparison
data1 = s1.serialize()
data2 = s2.serialize()

result = compare_summaries(data1, data2)
print(f"Similarity: {result.composite_similarity:.3f}")
print(f"Topology overlap: {result.overlap_score:.3f}")
print(f"Betti distance: {result.betti_distance:.3f}")
```


[Back to index](index.md)
