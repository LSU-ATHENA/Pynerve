# Runtime Validation & Formal Properties

## Validation at runtime

Pynerve performs runtime validation in debug builds and when `validate_results=true`:

```cpp
if (config.validate_results) {
    // Verify result invariants
    NERVE_ASSERT(pairs_sorted_by_dimension(pairs));
    NERVE_ASSERT(all_deaths_after_births(pairs));
    NERVE_ASSERT(betti_numbers_consistent(pairs));

    // Verify boundary matrix invariants
    NERVE_ASSERT(matrix_is_in_reduced_form(reduced));
    NERVE_ASSERT(lowest_one_correct_pairs(pairs, reduced));

    // Verify determinism contract
    if (config.require_bitwise_reproducibility) {
        NERVE_ASSERT(checksum == expected_checksum);
    }
}
```

These assertions are compiled out in Release builds.


## Formal properties

Pynerve guarantees the following formal properties of the computed persistence diagram:

1. **Persistence pairing is a perfect matching**: Every simplex is either paired (birth and death both finite) or essential (birth only, infinite death)
2. **Filtration order is preserved**: If simplex A is added before simplex B, then either birth(A) < birth(B) or A is paired with death > B's birth
3. **Zero-dimensional pairing is correct**: H0 births correspond to connected components, deaths correspond to component mergers
4. **Betti numbers are correct**: The number of pairs at each dimension equals the Betti number for that dimension
5. **Stability under perturbation**: Small perturbations to the input produce small changes in bottleneck distance (Lipschitz constant = 1 for bottleneck)

Property 5 holds only when the same determinism configuration is used for both computations.


[Back to Correctness Index](index.md)
