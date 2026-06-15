# When to use

Use the witness complex in the following scenarios. For datasets with $n > 50K$ in any dimension, use the witness complex with farthest-point landmarks. For $n > 100K$ with $\dim \leq 5$, use the witness complex with $500$ to $1000$ landmarks. When an approximate result is acceptable, use the witness complex with a small $m$ for speed. For streaming or distributed settings, run the witness complex independently on each chunk. When the data has high ambient dimension but low intrinsic dimension, the witness complex exploits the low intrinsic structure. For noisy data with many outliers, farthest-point landmark selection is recommended as it ignores isolated outliers.

<- [Witness Complex Overview](index.md)
