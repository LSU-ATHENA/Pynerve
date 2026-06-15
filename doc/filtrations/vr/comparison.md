# Comparison to other filtrations

The four filtrations --- VR, Alpha, Witness, and Sparse VR --- differ across several properties. Regarding metric support, VR, Witness, and Sparse VR accept any metric, while Alpha requires Euclidean only. For dimension constraints, VR, Witness, and Sparse VR have none, while Alpha requires dimension at most 3. VR and Alpha produce exact results (Alpha is homotopy-equivalent), whereas Witness gives a 3-times approximation and Sparse VR gives a $\frac{1}{1-\varepsilon}$ approximation. VR supports differentiability; Alpha does not; Witness has limited support; Sparse VR does not. Filtration size for VR is $O(n^{k+1})$, for Alpha $O(n^{\lceil d/2 \rceil})$, and for both Witness and Sparse VR $O(m^{k+1})$. Memory usage is high for VR, low for Alpha, moderate for Witness, and low for Sparse VR. VR is best for up to $10^4$ points, Alpha for up to $10^6$ points (with dimension at most 3), Witness for more than $10^5$ points, and Sparse VR for more than $10^4$ points.


<- [Vietoris-Rips Overview](index.md)
