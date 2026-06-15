# References

[Back to Index](index.md)

### Foundational Papers

- de Silva, V., Morozov, D., Vejdemo-Johansson, M. (2011). Dualities in Persistent Cohomology. *Inverse Problems*, 27(12), 124003. [Establishes the cohomology framework for persistence and the reverse-ordering algorithm.]

- de Silva, V., Morozov, D. (2011). Persistent Cohomology and Circular Coordinates. *Discrete & Computational Geometry*, 45(4), 737-759. [Develops the emergent pair detection and demonstrates cohomology's computational advantages.]

### Implementation Papers

- Bauer, U. (2021). Ripser: Efficient Computation of Vietoris-Rips Persistence Barcodes. *Journal of Applied and Computational Topology*, 5(3), 391-423. [Describes the optimized cohomology reduction used in Ripser, including implicit matrix representation and SIMD acceleration.]

- Maria, C. et al. (2014). The Gudhi Library: Simplicial Complexes and Persistent Homology. *HPCS 2014*, 167-174. [Describes the Gudhi library's implementation of cohomology persistence.]

- Bauer, U., Kerber, M., Reininghaus, J. (2014). Distributed Computation of Persistent Homology. *ALENEX 2014*, 31-40. [Describes distributed cohomology computation strategies.]

### GPU and Parallel Methods

- Zhang, S., Guo, H., Bhatt, S., Shen, H.-W., Peterka, T. (2020). A GPU-Accelerated Persistent Homology Algorithm. *Euro-Par 2020*, 285-300. [Describes warp-level cohomology reduction on CUDA.]

- Ha, L., Xu, J., Peterka, T., Shen, H.-W. (2020). A CUDA-Accelerated Algorithm for Computing Persistent Homology. *PPoPP 2020*, 235-246. [Alternative GPU approach using fine-grained parallelism.]

### Survey and Comparison

- Otter, N., Porter, M.A., Tillmann, U., Grindrod, P., Harrington, H.A. (2017). A Roadmap for the Computation of Persistent Homology. *EPJ Data Science*, 6(1), 17. [Comprehensive survey comparing homology and cohomology approaches.]

- Boissonnat, J.-D., Chazal, F., Yvinec, M. (2018). *Geometric and Topological Inference*. Cambridge University Press. [Covers the geometric aspects of persistence computation with cohomology.]
