from pyarrow import csv

def pyarrow_load(path):
    pyarrow_df = csv.read_csv(path)
    return pyarrow_df.to_pandas()

if __name__ == "__main__":
    df = pyarrow_load('week7/2023_01.csv')
    print(df)