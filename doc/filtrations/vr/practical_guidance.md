# Practical guidance: when to use VR

Choose your approach based on the problem scale and requirements. For fewer than a thousand points with dimension at most 3, use exact VR with the PH4 engine. For one thousand to ten thousand points with dimension at most 3, use exact VR with the cohomology engine. For ten thousand to one hundred thousand points, use sparse VR or the witness complex. Beyond one hundred thousand points, use sparse VR with streaming. For high-dimensional data where dimension exceeds 10, prefer the witness complex. For geometric data with dimension at most 3, the alpha complex produces a smaller, faster filtration. For differentiable topology, use VR with PyTorch autograd. For exploratory analysis, use sparse VR with epsilon between 0.4 and 0.5. For production deployment, use PH4 with memory mode set to streaming.

### When NOT to use VR

There are several scenarios where alternative filtrations are more appropriate. For exact topology on low-dimensional geometric data, the alpha complex produces a filtration 10 to 100 times smaller. For more than one hundred thousand points with limited RAM, use sparse VR or the witness complex. For points on a regular grid, the cubical complex is more appropriate (see level_set.md). If only connected components are needed, single-linkage clustering is faster. For real-time or streaming settings, use sparse VR on chunks of data.


<- [Vietoris-Rips Overview](index.md)
