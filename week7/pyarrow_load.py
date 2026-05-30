from pyarrow import csv

def pyarrow_load(path):
    return csv.read_csv(path)

if __name__ == "__main__":
    df = pyarrow_load('week7/2023_01.csv')
    print(df)