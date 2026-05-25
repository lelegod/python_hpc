# numactl --interleave=all

## What it does
Spreads memory allocation across all NUMA nodes (CPU sockets) in round-robin.
Does NOT control which cores run the code — only where memory is allocated.

## Why it matters on HPC (XeonGold6226R)
The node has 2 physical CPU sockets, each with its own local RAM (NUMA architecture).
- Without numactl: all memory allocated on socket 0
  - Cores on socket 0 → fast local access
  - Cores on socket 1 → slow remote access over inter-socket link
  - Result: speedup grows up to ~50% of cores, then plateaus or drops
- With numactl --interleave=all: memory spread across both sockets
  - All cores get roughly equal access times
  - Result: speedup scales with every core added

## Usage
    numactl --interleave=all python my_script.py

## Quiz answers
1. Before numactl: speedup improves until 50% of threads added, then decreases
2. After numactl:  speedup now increases with every core added
3. Effect:         ensures memory allocation is spread to both CPU sockets
