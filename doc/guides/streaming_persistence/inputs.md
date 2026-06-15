# Input sources

Pynerve supports several input sources. NPY files (`.npy`) provide full arrays chunked internally. NPZ files (`.npz`) use the first array or the `"data"` key. HDF5 files (`.h5` / `.hdf5`) require `h5py`. Async iterators (`AsyncIterator[ndarray]`) serve as custom data sources. Memory streams (`DataStream`, C++ only) support network or shared memory.

```python
# Custom async source
async def my_data_source():
    for frame in my_sensor_stream():
        yield np.asarray(frame)

sp = StreamingPersistence(chunk_size=500, max_dim=1)
async for result in sp.stream_compute(my_data_source()):
    print(result)
```

### HDF5 chunking

When reading from HDF5, Pynerve uses the dataset's native chunk layout:

```python
# HDF5 with chunk cache tuning
async for result in sp.stream_compute(
    "data.h5",
    chunk_size=2048,
    # HDF5 chunk cache: hundreds of megabytes
    # hdf5_chunk_cache_mb=100,
):
    pass
```

[Back to index](index.md)
