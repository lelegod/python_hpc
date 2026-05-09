import sys
import pandas as pd
import numpy as np

def pandas_chunks(path_name, size):
    df = pd.read_csv(path_name, nrows = size)
    print(np.sum(df.value))

if __name__ == '__main__':
    # path_name = str(sys.argv[1])
    # size = int(sys.argv[2])
    path_name = "/dtu/projects/02613_2025/data/dmi/2023_01.csv.zip"
    size = 5
    pandas_chunks(path_name, size)