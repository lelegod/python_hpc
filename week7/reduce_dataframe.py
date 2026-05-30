import pandas as pd

def summarize_columns(df):
    print(pd.DataFrame([
        (
            c,
            df[c].dtype,
            len(df[c].unique()),
            df[c].memory_usage(deep=True) // (1024**2)
        ) for c in df.columns
    ], columns=['name', 'dtype', 'unique', 'size (MB)']))
    print('Total size:', df.memory_usage(deep=True).sum() / 1024**2, 'MB')

def reduce_dmi_df(df):
    for column in df.columns:
        if column in ("created", "observed"):
            df[column] = pd.to_datetime(df[column], format="ISO8601")
        elif pd.api.types.is_integer_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], downcast="integer")
        elif pd.api.types.is_float_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], downcast="float")
        elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype("category")
    return df

if __name__ == '__main__':
    df = reduce_dmi_df(pd.read_csv('week7/2023_01.csv.zip'))
    print(summarize_columns(df))
