import sys
import os
import blosc
from time import perf_counter
import numpy as np
from functools import wraps

def time_it(func):
    @wraps(func)
    def wrapper(*args):
        start = perf_counter()
        result = func(*args)
        end = perf_counter()
        print(f"{end - start}")
        return result
    return wrapper

@time_it
def write_numpy(arr, file_name):
    np.save(f"{file_name}.npy", arr)
    if hasattr(os, 'sync'):
        os.sync()

@time_it
def write_blosc(arr, file_name, cname="lz4"):
    b_arr = blosc.pack_array(arr, cname=cname)
    with open(f"{file_name}.bl", "wb") as w:
        w.write(b_arr)
    if hasattr(os, 'sync'):
        os.sync()

@time_it
def read_numpy(file_name):
    return np.load(f"{file_name}.npy")

@time_it
def read_blosc(file_name):
    with open(f"{file_name}.bl", "rb") as r:
        b_arr = r.read()
    return blosc.unpack_array(b_arr)

def write_read(A):
    write_numpy(A, 'write')
    write_blosc(A, 'write')
    read_numpy('write')
    read_blosc('write')

def zero(n):
    A = np.zeros((n, n, n), dtype='uint8')
    print(f"Zero ({n}): ")
    write_read(A)

def tiled(n):
    tiled_array = np.tile(
        np.arange(256, dtype='uint8'),
        (n // 256) * n * n,
        ).reshape(n, n, n)
    print(f"Tiled ({n}): ")
    write_read(tiled_array)

def random(n):
    random_array = np.random.randint(low=1, high=256, size=(n, n, n), dtype='uint8')
    print(f"Random ({n}): ")
    write_read(random_array)

if __name__ == '__main__':
    n_list = [256, 512, 1024]
    for n in n_list:
        zero(n)
        tiled(n)
        random(n)