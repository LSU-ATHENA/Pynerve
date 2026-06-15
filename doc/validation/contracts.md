## API Contracts

API signature validation between C++ and Python. Ensures that function
signatures, parameter types, and return types match across the pybind11
binding layer.

```python
from pynerve.validation.contracts import (
    validate_python_signature,
    validate_cpp_signature,
    validate_binding_consistency,
    ValidationReport,
)

report = validate_binding_consistency("pynerve.algebra.build_vr_complex")
# report.signature_match: bool
# report.python_params: list[str]
# report.cpp_params: list[str]
# report.type_mismatches: list[str]
# report.missing_params: list[str]
# report.extra_params: list[str]
```

### Per-function validation

```python
sig = validate_python_signature("pynerve.sheaf.SheafLaplacian.build_sheaf_laplacian")
# sig.name, sig.params, sig.return_type, sig.docstring

sig = validate_cpp_signature("_ZN5pynerve...")
# sig.name, sig.params, sig.return_type

report = validate_binding_consistency("pynerve.spectral.Laplacian.eigenvalues")
print(report.to_json())
```

### Batch validation

```python
from pynerve.validation.contracts import validate_all_bindings

report = validate_all_bindings()
# report.total_functions: int
# report.passed: int
# report.failed: int
# report.details: list[ValidationReport]
```


## How contract validation works

The validation system inspects:

1. **Python side**: Uses `inspect.signature()` to extract parameter names, types, defaults, and return annotations
2. **C++ side**: Parses mangled names via `c++filt` or reads debug info from compiled symbols
3. **Binding layer**: Cross-references pybind11 registration signatures with both sides

```python
report = validate_binding_consistency("pynerve.algebra.build_vr_complex")
print(f"Signature match: {report.signature_match}")
print(f"Python params: {report.python_params}")
print(f"C++ params: {report.cpp_params}")

if report.type_mismatches:
    for m in report.type_mismatches:
        print(f"Type mismatch on '{m.param}': "
              f"Python says {m.py_type}, C++ says {m.cpp_type}")
```

### Typical mismatches detected

Common binding mismatches fall into four categories:

- **Optional parameter missing** -- Python provides a default (`max_radius=inf`) but C++ requires `float max_radius`. Fix by adding a default to C++ or using `py::arg`.
- **Type width mismatch** -- Python uses `int` but C++ expects `size_t`. Fix by adding a pybind11 type cast.
- **Renamed parameter** -- Python uses `max_dim` but C++ uses `max_dimension`. Fix by aligning naming or adding `py::arg("max_dim")`.
- **Wrong parameter order** -- Python passes `(points, dim)` but C++ expects `(dim, points)`. Fix by reordering or using keyword-only arguments.

### Adding new bindings

When adding a new pybind11 binding, the validation will automatically detect it in the next `validate_all_bindings()` run. To ensure clean validation:

```cpp
// pybind11 binding with explicit arg names and docs
m.def("my_function", &my_function,
    py::arg("points"), py::arg("max_dim") = 2,
    "Compute my custom topology feature");
```

Then verify:

```python
report = validate_binding_consistency("pynerve.my_function")
assert report.signature_match, f"Binding mismatch: {report}"
```

### CI enforcement

```bash
# In CI: validate all bindings, fail if any mismatch
python -c "
from pynerve.validation.contracts import validate_all_bindings
r = validate_all_bindings()
assert r.passed == r.total_functions, f'{r.failed} binding mismatches!'
for d in r.details:
    if not d.signature_match:
        print(f'FAIL: {d.function}')
print(f'OK: {r.passed}/{r.total_functions} signatures validated')
"
```

### Custom validation rules

```python
from pynerve.validation.contracts import BindingValidator, ValidationRule

class NoMutableDefaults(BindingValidationRule):
    def check(self, py_sig, cpp_sig):
        for p in py_sig.params:
            if isinstance(p.default, (list, dict, set)):
                return ValidationError(f"Mutable default in '{p.name}'")
        return None

validator = BindingValidator()
validator.add_rule(NoMutableDefaults())
report = validator.validate("pynerve.algebra.build_vr_complex")
```


## FAQ

**Q: Do I need to run contract validation after every change?**
A: At minimum, run before committing any change to pybind11 bindings. The validation catches silent mismatches that would cause runtime TypeError.

**Q: What happens on a type mismatch?**
A: If Python passes `int` but C++ expects `size_t`, pybind11 performs implicit conversion and succeeds. But if Python passes `float` and C++ expects `Tensor`, the binding will throw at runtime. Contract validation catches these before runtime.

**Q: Can I skip validation for internal functions?**
A: Yes. Internal C++ functions not exposed to Python are skipped. Only functions registered via `m.def()` or `m.export_values()` are validated against Python signatures.


### Cross-references

- `pynerve.validation`: Validation overview
- `pynerve.validation.determinism`: Output determinism
- `pynerve.validation.launch_audit`: CUDA launch validation
- `pynerve.cmake`: CMake pybind11 registration macros
