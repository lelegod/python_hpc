import pandas as pd

def df_memsize(df: pd.DataFrame):
    return df.memory_usage(deep=True).sum()

if __name__ == '__main__':
    df = pd.read_csv('week7/2023_01.csv.zip')
    print(df_memsize(df))
