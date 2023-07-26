import pandas as pd

def add_col_from_params(df: pd.DataFrame, col_name, col_value, param_name):
    df.loc[df['parameters'].str.contains(param_name, regex=False), col_name] = col_value
    return df