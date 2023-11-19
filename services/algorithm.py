import os
import pickle
from types import NoneType
from typing import Tuple, Any, Literal, List
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def create_model(samples: dict[Literal['x', 'y'], Any]) -> Tuple[LinearRegression, Any, Any]:
    x = samples['x'].values
    y = samples['y'].values
    regressor = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor.fit(x_train, y_train)
    return regressor, x_test, y_test


def save_model(regressor: LinearRegression, filename: str) -> None:
    _path = Path('lr_models')
    _path.mkdir(exist_ok=True)
    pickle.dump(regressor, open(f'lr_models/{filename}', "wb"))


def load_model(filename: str | None = None) -> LinearRegression:
    if isinstance(filename, NoneType):
        files = os.listdir('lr_models')
        if files:
            filename = sorted(files)[0]

    return pickle.load(open(f'lr_models/{filename}', "rb"))


def get_coefficients(regressor: LinearRegression, x_columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(regressor.coef_, x_columns, columns=['Коэффициент'])


def get_predict(regressor: LinearRegression, input_x: Any) -> Any:
    return regressor.predict(input_x)


def check_model(regressor: LinearRegression, x_test: Any, y_test: Any) -> bool:
    y_predicted = get_predict(regressor, x_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_predicted))
    print('Средняя абсолютная ошибка:', metrics.mean_absolute_error(y_test, y_predicted))
    print('Средняя квадратическая ошибка:', metrics.mean_squared_error(y_test, y_predicted))
    print('Среднеквадратичная ошибка:', np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
    print(y_test.mean())
    return rmse / y_test.mean() < 0.1
