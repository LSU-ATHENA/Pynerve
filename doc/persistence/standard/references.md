# References

### Foundational Papers

- Zomorodian, A., Carlsson, G. (2005). Computing Persistent Homology. *Discrete & Computational Geometry*, 33(2), 249-274. [Establishes the standard reduction algorithm over arbitrary fields.]

- Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological Persistence and Simplification. *Discrete & Computational Geometry*, 28(4), 511-533. [Original persistence algorithm for 3D functions.]

### Optimization Papers

- Chen, C., Kerber, M. (2011). Persistent Homology Computation with a Twist. *Proc. 27th European Workshop on Computational Geometry*, 11-14. [Introduces the clearing optimization.]

- Bauer, U. (2021). Ripser: Efficient Computation of Vietoris-Rips Persistence Barcodes. *Journal of Applied and Computational Topology*, 5(3), 391-423. [Describes compression, implicit matrix representation, and practical optimization techniques.]

### Implementation and Survey

- Otter, N., Porter, M.A., Tillmann, U., Grindrod, P., Harrington, H.A. (2017). A Roadmap for the Computation of Persistent Homology. *EPJ Data Science*, 6(1), 17. [Survey of algorithms and implementations.]

- Edelsbrunner, H., Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society. [Textbook covering the mathematical foundations.]

- Boissonnat, J.-D., Chazal, F., Yvinec, M. (2018). *Geometric and Topological Inference*. Cambridge University Press. [Covers the geometric aspects of persistence computation.]

### Historical Notes

The standard reduction algorithm is sometimes called the "column reduction algorithm" or "Zomorodian-Carlsson algorithm." It was the first practical algorithm for computing persistent homology over arbitrary fields and remains the most widely taught and understood method. Its O(n^3) worst-case complexity is acceptable for the moderate-sized complexes common in early TDA applications (n < 10^4), but modern applications routinely require n > 10^6, motivating the cohomology-based approaches described in the companion documentation.

<- [Standard Reduction Overview](index.md)
