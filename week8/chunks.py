import sys
import pandas as pd
import numpy as np

def pandas_chunks(path_name, size):
    total = 0
    for chunk in pd.read_csv(path_name, chunksize=size):
        total += chunk.loc[chunk['parameterId'] == 'precip_past10min', 'value'].sum()
    print(total)

if __name__ == '__main__':
    path_name = str(sys.argv[1])
    size = int(sys.argv[2])
    pandas_chunks(path_name, size)