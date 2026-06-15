# Python exception wrapping

pybind11 type caster in `python/bindings/detail/nerve_torch_bindings_exceptions.inl`
translates C++ `nerve::errors::NerveError` exceptions to the corresponding
Python `NerveError` subclass:

```
C++                              Python
nerve::errors::NerveError   ->   NerveError
nerve::errors::GPUError     ->   GPUError
nerve::errors::PersistenceError -> PersistenceError
(nested error_code)              (e.error_code attribute)
```

Python `translate_cpp_exception()` provides manual translation for
callbacks and custom binding points.


[Back to index](index.md)
