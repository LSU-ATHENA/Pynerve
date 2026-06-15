# Checkpoint/restart

The sharded boundary matrix supports checkpointing for fault tolerance:

```python
result = pynerve.distributed_persistence(
    points,
    max_dim=2,
    checkpoint_path="/scratch/nerve_ckpt/",
    checkpoint_interval_sec=300,
)
```

Each rank checkpoints its local column state to `checkpoint_path/rank_{id}/`. On restart, `restore()` reloads the state and resumes from the last consistent global state.

### Checkpoint format

```
checkpoint_path/
  rank_0/
    columns_0001.bin     -- serialized columns (up to 10K entries)
    columns_0002.bin
    metadata.json        -- rank, world_size, checkpoint_id, timestamp
  rank_1/
    columns_0001.bin
    metadata.json
  ...
  global_state.json      -- consistent global state marker
```

### Restart protocol

```
1. Each rank reads its latest checkpoint:
   state = restore(checkpoint_path/rank_{my_rank}/)

2. Rank 0 broadcasts the highest consistent checkpoint_id:
   MPI_Bcast(&checkpoint_id, 1, MPI_INT, 0, MPI_COMM_WORLD)

3. All ranks that have checkpoint_id in their metadata load it.
   Ranks without the checkpoint_id recompute from scratch.

4. Global barrier synchronizes after restore.
```

<- [Distributed Computing Overview](index.md)
