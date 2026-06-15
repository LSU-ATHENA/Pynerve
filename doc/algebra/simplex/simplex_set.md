# SimplexSet

```cpp
class SimplexSet {
public:
    bool insert(const Simplex& s);
    bool erase(const Simplex& s);
    bool contains(const Simplex& s) const;
    Size size() const noexcept;
    bool empty() const noexcept;
    void clear();

    SimplexSet intersection(const SimplexSet& other) const;
    SimplexSet unionSet(const SimplexSet& other) const;
    SimplexSet difference(const SimplexSet& other) const;
    SimplexSet kSimplices(Size k) const;
    Size numKSimplices(Size k) const;
    Size maxDimension() const;
    SimplexSet boundary() const;
    SimplexSet kBoundary(Size k) const;
    SimplexSet star(const Simplex& simplex) const;
    SimplexSet link(const Simplex& simplex) const;
    std::vector<Simplex> toVector() const;
};
```

Backed by `unordered_set<Simplex>`. Use for fast membership queries and
set algebra on collections of simplices.

### Set operations

`intersection` returns simplices in both sets in O(min(a,b)). `unionSet` returns simplices in either set in O(a + b). `difference` returns simplices in the first but not the second in O(a). `kSimplices` filters by dimension in O(n). `boundary` computes all boundary simplices of all members in O(n * f). `star(s)` returns all cofaces of s in the set in O(n). `link(s)` returns the star minus those containing s as a face in O(n).

```python
from pynerve.algebra import Simplex, SimplexSet

s1 = SimplexSet()
s1.insert(Simplex([0, 1, 2]))
s1.insert(Simplex([0, 1]))

s2 = SimplexSet()
s2.insert(Simplex([0, 1]))
s2.insert(Simplex([1, 2]))

common = s1.intersection(s2)   # {[0,1]}
all_s = s1.unionSet(s2)        # {[0,1,2], [0,1], [1,2]}
diff = s1.difference(s2)       # {[0,1,2]}

edges = s1.kSimplices(1)       # {[0,1]}
```


[Back to index](index.md)
