# Work-stealing scheduler

Load imbalance is handled by a distributed work-stealing scheduler:

```cpp
// src/distributed/work_stealing_scheduler.cpp
class WorkStealingScheduler {
    void submit_work(std::function<void()> task);
    void run();
    bool steal_work(std::function<void()>& stolen);
    void shutdown();
};
```

When a rank exhausts its local columns, it attempts to steal work from a random peer. Failed steals trigger exponential backoff (1ms -> 2ms -> 4ms -> ... capped at 1s).

### Work-stealing protocol

```
1. Rank A finishes all local columns.
2. Rank A selects a random victim rank V (V != A).
3. Rank A sends MPI probe (TAG_WORK_STEAL) to Rank V.
4. Rank V:
   a. If V has pending work:
      - Serialize the next pending column into a buffer
      - MPI_Send(buffer, size, MPI_UNSIGNED_LONG_LONG, A, TAG_WORK_STEAL)
   b. If V has no pending work:
      - MPI_Send(empty_flag, 1, MPI_CHAR, A, TAG_WORK_STEAL)
5. Rank A:
   a. MPI_Probe to determine message size
   b. MPI_Recv the work or empty flag
   c. If work received: reduce the stolen column
   d. If empty: exponential backoff, then try new victim
6. If no work available after trying all peers, rank goes idle.
```

### Backoff strategy

The backoff strategy doubles at each attempt: attempt 1 waits 1 ms, attempt 2 waits 2 ms, attempt 3 waits 4 ms, attempt 4 waits 8 ms, and attempts 5 and beyond wait 16 ms (capped at 1 s).

After 10 consecutive failed steals, the rank enters an idle state and polls for new work at 100 ms intervals.

<- [Distributed Computing Overview](index.md)
