# Definition

[Back to index](index.md)

The sparse VR complex is built in three steps:

1. **Epsilon-net.** Select a set of landmark points $L \subset X$ via greedy
   farthest-point sampling (maxmin), where each successive landmark is at
   distance at least $\delta$ from all previous landmarks.
2. **Sparse distance.** Define a scaled distance $d_L$ on the landmarks that
   $(1+\varepsilon)$-approximates the ambient distance.
3. **Filtration.** Build the VR complex on $L$ using $d_L$, producing a
   filtration $\varepsilon$-interleaved with the true VR on $X$.
