from typing import Literal, List, Tuple, Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_data(path: str, date_cols: List[str], date_format: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=date_cols, date_format=date_format)


def drop_na(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.dropna(subset=columns)


def fill_na(df: pd.DataFrame, column: str, _type: Literal['mean', 'mode']) -> pd.DataFrame:
    if _type == 'mean':
        df[column] = df[column].fillna(df[column].mean())
    else:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df


def wash_invalid_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return df[(df[column] > (q1 - 1.5 * iqr)) & (df[column] < (q3 + 1.5 * iqr))]


def label_values(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    le.fit(df[column])
    df[column] = le.transform(df[column])
    return df


def create_samples(df: pd.DataFrame, x_columns: List[str], y_column: str) -> Tuple[Any, Any]:
    return df[x_columns], df[y_column]
