# Performance analysis

[Back to index](index.md)

### Scaling with epsilon

Measured on $n = 100,000$ points in 3D, $\max\\_dim = 2$:

With $\varepsilon = 0.1$ and 10,000 landmarks, construction takes tens of seconds and reduction tens of seconds, finding roughly 11,000 pairs while using over a gigabyte of memory. At $\varepsilon = 0.2$ and 4,000 landmarks, construction completes in seconds and reduction in seconds, finding around 8,500 pairs with hundreds of megabytes. The default $\varepsilon = 0.3$ with 2,000 landmarks finishes construction and reduction in seconds each, finding about 6,200 pairs using hundreds of megabytes. With $\varepsilon = 0.4$ and 1,200 landmarks, both construction and reduction finish in roughly a second, finding about 4,800 pairs with tens of megabytes. At the loosest setting $\varepsilon = 0.5$ and 800 landmarks, construction and reduction are sub-second, finding roughly 3,500 pairs and using tens of megabytes.

### Scaling with point count

Fixed $\varepsilon = 0.3$, $\max\\_dim = 2$:

With 10,000 points and 300 landmarks, total time is under a second and memory usage is in the tens of megabytes. At 50,000 points and 1,200 landmarks, time is a few seconds and memory remains in the tens of megabytes. With 100,000 points and 2,000 landmarks, total time is a few seconds and memory reaches hundreds of megabytes. At 500,000 points and 6,000 landmarks, time is tens of seconds and memory is hundreds of megabytes. With 1,000,000 points and 10,000 landmarks, total time is around two minutes and memory is over a gigabyte.

The sparse VR memory grows **linearly** with $n$ (via $m$ growth), not
exponentially as in exact VR.

### Approximation quality vs epsilon

Using bottleneck distance to exact VR on a 5,000-point subset:

At $\varepsilon = 0.1$, the theoretical bound of 0.105 is conservative against an observed bottleneck of 0.08, giving an observed-to-theory ratio of 0.76. With $\varepsilon = 0.2$, the theoretical bound of 0.223 compares to an observed 0.15, ratio 0.67. At the default $\varepsilon = 0.3$, theory gives 0.357 versus an observed 0.22, ratio 0.62. With $\varepsilon = 0.4$, the bound is 0.511 versus 0.30 observed, ratio 0.59. At $\varepsilon = 0.5$, theory gives 0.693 versus 0.38 observed, ratio 0.55.

Observed error is typically 40-60% of the theoretical bound for random
data. Worst-case behavior is rare in practice.
