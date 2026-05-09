import sys
import pandas as pd

def pandas_chunks(path_name, size):
    df = pd.read_csv(path_name, nrows = size)
    print(df)

if __name__ == '__main__':
    path_name = str(sys.argv(1))
    size = int(sys.argv(2))
    pandas_chunks()