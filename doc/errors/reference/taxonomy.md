# Error code taxonomy

Error codes are 32-bit integers with component-specific hex ranges. The IO / infrastructure range `0x000001xx` includes codes like `E00_IO_TIMEOUT`. The GPU compute range `0x000002xx` covers `E10_GPU_OOM` and `E11_GPU_LAUNCH_FAIL`. The numerical range `0x000003xx` includes `E20_NUM_NAN` and `E21_NUM_NO_CONVERGE`. The determinism range `0x000004xx` covers `E30_DET_MISMATCH` and `E31_SCHEMA_VERSION`. The capacity / resource range `0x000005xx` includes `E40_CPU_OVERLOAD` and `E41_RESOURCE_LIMIT`. The algorithmic / PH range `0x000006xx` covers `E50_PH_ABORT` through `E13_PH_HIGHDIM_PRECISION`. The NUMA / affinity range `0x000007xx` includes `E60_NUMA_BIND_FAIL` through `E62_NUMA_MIGRATION_ERROR`. The precision range `0x000008xx` covers `E70_PRECISION_DOWNGRADE` through `E73_PRECISION_CATASTROPHIC`. The matrix / homology range `0x000009xx` includes `E81_MATRIX_EMPTY` through `E94_CONVERGENCE_FAILURE`.


[Back to index](index.md)
