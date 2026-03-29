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

def main():
    n = int(sys.argv[1])
    A = np.zeros((n, n, n), dtype='uint8')

    write_numpy(A, 'write')
    write_blosc(A, 'write')
    read_numpy('write')
    read_blosc('write')

if __name__ == '__main__':
    main()