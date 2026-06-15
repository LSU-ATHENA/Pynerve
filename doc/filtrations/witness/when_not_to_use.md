# When not to use

Do not use the witness complex in these scenarios. When $n < 10K$, use exact VR instead as it is fast and exact. For geometric data with $\dim \leq 3$, use the Alpha complex which is smaller and exact. When high precision is required, use VR with tolerances or sparse VR with guarantees. If the data is highly non-uniform, sparse VR may give a better approximation.

<- [Witness Complex Overview](index.md)
