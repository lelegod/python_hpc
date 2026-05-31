import numpy as np
from numba import cuda
from time import perf_counter

@cuda.jit
def add_kernel(x, y, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        out[i] = x[i] + y[i]

def main():
    n = 1_000_000
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(x.dtype)
    out = np.empty_like(x)

    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock

    # Warmup — ensures kernel is JIT compiled before timing
    add_kernel[blockspergrid, threadsperblock](x, y, out)
    cuda.synchronize()

    rep = 200
    t = perf_counter()
    for _ in range(rep):
        add_kernel[blockspergrid, threadsperblock](x, y, out)
    cuda.synchronize()
    print((perf_counter() - t) / rep * 1000, 'ms')

if __name__ == '__main__':
    main()
