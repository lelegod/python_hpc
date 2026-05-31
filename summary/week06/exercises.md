# Week 6 Exercises — Parallelism Part 2: Reductions, Shared Memory, numactl

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Background: Mean Faces](#background-mean-faces)
- [Exercise 1 `[AUTOLAB]`](#exercise-1-autolab)
- [Exercise 2 `[AUTOLAB]`](#exercise-2-autolab)
- [Exercise 3 `[PRACTICE]`](#exercise-3-practice)
- [Exercise 4 `[PRACTICE]`](#exercise-4-practice)
- [Exercise 5 `[AUTOLAB]`](#exercise-5-autolab)
- [Exercise 6 `[PRACTICE]`](#exercise-6-practice)

---

---

## Background: Mean Faces

The exercises use the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset of celebrity images. It is located at `/dtu/projects/02613_2025/data/celeba/`. Smaller subsets are available for testing (`celeba_200.npy`, `celeba_100K.npy`).

Each image is 128×128 RGB pixels, stored in an N×128×128×3 float32 array. The goal is to compute the **mean face** by summing all images and dividing by N using a **parallel reduction over shared memory**, minimising data transfers between processes.

**Template code (provided in the exercise):**

```python
import ctypes
import multiprocessing as mp
import sys
from time import perf_counter as time
import numpy as np
from PIL import Image


def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_


def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr, dtype='float32')


def reduce_step(args):
    b, e, s, elemshape = args
    arr = tonumpyarray(shared_arr).reshape((-1,) + elemshape)
    # Change the code below to compute a step of the reduction
    # ---------------------------8<---------------------------
    arr[b:e:s] = 1.0 - arr[b:e:s]  # <-- Dummy op. Replace with correct


if __name__ == '__main__':
    n_processes = 1
    chunk = 2

    # Create shared array
    data = np.load(sys.argv[1])
    elemshape = data.shape[1:]
    shared_arr = mp.RawArray(ctypes.c_float, data.size)
    arr = tonumpyarray(shared_arr).reshape(data.shape)
    np.copyto(arr, data)
    del data

    # Run parallel sum
    t = time()
    pool = mp.Pool(n_processes, initializer=init, initargs=(shared_arr,))

    # Change the code below to compute a step of the reduction
    # ---------------------------8<---------------------------
    pool.map(reduce_step,
             [(i, i + chunk, 1, elemshape) for i in range(0, len(arr), chunk)],
             chunksize=1)

    # Write output
    print(time() - t)
    final_image = arr[0]
    # final_image /= len(arr) # For mean
    Image.fromarray(
        (255 * final_image.astype(float)).astype('uint8')
    ).save('result.png')
```

---

## Exercise 1 `[AUTOLAB]`

Modify the provided example to compute the first step of a parallel reduction, i.e., every second element is summed with its neighbor. For example, after this step, an input of [1, 2, 3, 4] should be [1+2, 2, 3+4, 4] = [3, 2, 7, 4].

Hint: for testing, make some dummy data like:

```python
import numpy as np
arr = np.arange(10)  # 0 to 9
arr = arr.astype('float32')
arr = arr[:, None, None, None]  # (10, 1, 1, 1)
np.save('dummydata.npy', arr)
```

Then, comment out the lines saving the image and instead just print the array to manually inspect the results.

> **Solution:**
>
> In `reduce_step`, replace the dummy op with a pairwise addition: element at index `b` accumulates the element at `b + 1`. In `__main__`, `pool.map` is called once with pairs `(i, i+chunk, 1, elemshape)` stepping through the array.
>
> ```python
> import ctypes
> import multiprocessing as mp
> import sys
> from time import perf_counter as time
> import numpy as np
> from PIL import Image
>
>
> def init(shared_arr_):
>     global shared_arr
>     shared_arr = shared_arr_
>
>
> def tonumpyarray(mp_arr):
>     return np.frombuffer(mp_arr, dtype='float32')
>
>
> def reduce_step(args):
>     b, e, s, elemshape = args
>     arr = tonumpyarray(shared_arr).reshape((-1,) + elemshape)
>     # ---------------------------8<---------------------------
>     if b + 1 < len(arr):
>         arr[b] += arr[b + 1]
>
>
> if __name__ == '__main__':
>     n_processes = 1
>     chunk = 2
>
>     # Create shared array
>     data = np.load(sys.argv[1])
>     elemshape = data.shape[1:]
>     shared_arr = mp.RawArray(ctypes.c_float, data.size)
>     arr = tonumpyarray(shared_arr).reshape(data.shape)
>     np.copyto(arr, data)
>     del data
>
>     # Run parallel sum
>     t = time()
>     pool = mp.Pool(n_processes, initializer=init, initargs=(shared_arr,))
>
>     # ---------------------------8<---------------------------
>     pool.map(reduce_step,
>              [(i, i + chunk, 1, elemshape) for i in range(0, len(arr), chunk)],
>              chunksize=1)
>
>     # Write output
>     print(time() - t)
>     final_image = arr[0]
>     # final_image /= len(arr) # For mean
>     Image.fromarray(
>         (255 * final_image.astype(float)).astype('uint8')
>     ).save('result.png')
> ```

---

## Exercise 2 `[AUTOLAB]`

Further modify the example to compute the full reduction. Remember to divide by the number of images at the end to compute the mean. Hint: each step of the reduction should be a separate call to `pool.map`.

> **Solution:**
>
> The full reduction requires ceil(log2(N)) rounds. At round j the stride `s = 2**j`, and each active element at index `i` adds the element `i + s` into itself. After all rounds, `arr[0]` holds the total sum; divide by `len(arr)` for the mean.
>
> ```python
> import ctypes
> import multiprocessing as mp
> import sys
> from time import perf_counter as time
> import numpy as np
> from PIL import Image
>
>
> def init(shared_arr_):
>     global shared_arr
>     shared_arr = shared_arr_
>
>
> def tonumpyarray(mp_arr):
>     return np.frombuffer(mp_arr, dtype='float32')
>
>
> def reduce_step(args):
>     b, e, s, elemshape = args
>     arr = tonumpyarray(shared_arr).reshape((-1,) + elemshape)
>     # ---------------------------8<---------------------------
>     if b + s < len(arr):
>         arr[b] += arr[b + s]
>
>
> if __name__ == '__main__':
>     n_processes = 1
>     chunk = 2
>
>     # Create shared array
>     data = np.load(sys.argv[1])
>     elemshape = data.shape[1:]
>     shared_arr = mp.RawArray(ctypes.c_float, data.size)
>     arr = tonumpyarray(shared_arr).reshape(data.shape)
>     np.copyto(arr, data)
>     del data
>
>     # Run parallel sum
>     t = time()
>     pool = mp.Pool(n_processes, initializer=init, initargs=(shared_arr,))
>
>     # ---------------------------8<---------------------------
>     for j in range(int(np.ceil(np.log2(len(arr))))):
>         s = 2**j
>         pool.map(reduce_step,
>                  [(i, 0, s, elemshape) for i in range(0, len(arr), 2*s)],
>                  chunksize=1)
>
>     # Write output
>     print(time() - t)
>     final_image = arr[0]
>     final_image /= len(arr)  # For mean
>     Image.fromarray(
>         (255 * final_image.astype(float)).astype('uint8')
>     ).save('result.png')
> ```

---

## Exercise 3 `[PRACTICE]`

Play with the chunk size (the `chunk` variable in the code). What value gives the best performance for you? Note, this will depend on the size of the subset you use, so make sure to try the program for at least the 100,000 subset.

> **Solution:**
>
> The optimal chunk size is hardware- and data-dependent. Experimentally, a chunk size of **64** gives good performance on the 100K subset. Larger chunks reduce scheduling overhead but may leave some cores idle at each reduction step; smaller chunks increase parallelism but add more pool.map overhead per round.

---

## Exercise 4 `[PRACTICE]`

Run your program as a batch job for varying number of processes and create a speedup plot. Run the program for at least the 100,000 subset. For consistent results, you should run the program 3 times and use the average time. What do you see? Is anything surprising? Hint: **this will take time** for the 100K subset or above. While you wait, start the next exercise.

> **Solution:**
>
> On a dual-socket node (e.g., Xeon Gold 6342 with 48 available threads), the speedup improves for the first ~24 cores (one socket), then **decreases** when adding cores from the second socket. This is surprising — we expect monotonically increasing speedup. The cause is the NUMA (Non-Uniform Memory Access) architecture: all memory is allocated on socket 0 by default, so cores on socket 1 pay a large latency penalty for every access over the inter-socket interconnect.

---

## Exercise 5 `[AUTOLAB]`

Repeat the previous exercise, but run Python under `numactl --interleave=all`. What changed? Hint: see lecture slides for examples with `numactl`.

> **Solution:**
>
> Run the program as:
>
> ```bash
> numactl --interleave=all python reduction.py <path/to/data.npy>
> ```
>
> `numactl --interleave=all` spreads memory allocation in round-robin across all NUMA nodes (both CPU sockets). It does **not** control which cores run the code — only where memory is allocated. With interleaved memory, all cores get roughly equal average memory access latency, so the speedup now increases with every core added (scaling continues beyond the first socket). The performance plateau or drop seen in Exercise 4 disappears.
>
> Key distinctions from the numactl notes:
> - **Without numactl:** all memory on socket 0; socket 1 cores see slow remote access; speedup stalls at ~50% of cores.
> - **With `--interleave=all`:** memory spread across both sockets; speedup scales with all cores.

---

## Exercise 6 `[PRACTICE]`

Compare your fastest runtime with `np.sum`. Do you manage to compute the sum faster? If so, by how much?

> **Solution:**
>
> The parallel reduction with numactl and optimal process count should outperform a single-threaded `np.sum`. However, the margin is modest: **2–4x speedup** is realistic. NumPy's `np.sum` is already highly optimised (using SIMD/vectorised C code), so it is very hard to significantly beat it. Python's multiprocessing introduces substantial per-round overhead (pool.map calls, argument serialisation), which limits the achievable speedup. Using the full parallel reduction is still worthwhile for very large datasets where the data-parallel speedup outweighs the fixed overhead.
