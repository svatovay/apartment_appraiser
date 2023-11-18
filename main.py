import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    # Чтение
    dataset = pd.read_csv('sold_flats_2020-09-30.csv', parse_dates=['date_sold'], date_format='%Y-%m-%d')

    # Нормализация
    dataset = dataset.dropna(subset=['sold_price', 'price'])
    sold_price_q = dataset['sold_price'].quantile(0.99)
    price_q = dataset['price'].quantile(0.99)
    dataset = dataset[dataset['sold_price'] < sold_price_q]
    dataset = dataset[dataset['price'] < price_q]
    dataset['area_total'] = dataset['area_total'].fillna(dataset['area_total'].mean())
    dataset['floor_num'] = dataset['floor_num'].fillna(dataset['floor_num'].mean())
    dataset['rooms_cnt'] = dataset['rooms_cnt'].fillna(dataset['rooms_cnt'].mean())
    X = dataset[
        ['city_id', 'district_id', 'street_id', 'price', 'area_total', 'floor_num', 'rooms_cnt']
    ]
    Y = dataset['sold_price']

    # Обучение
    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    # Проверка
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Коэффициент'])
    Y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Фактическая': Y_test, 'Предсказанная': Y_pred})

    print('Средняя абсолютная ошибка:', metrics.mean_absolute_error(Y_test, Y_pred))
    print('Средняя квадратическая ошибка:', metrics.mean_squared_error(Y_test, Y_pred))
    print('Среднеквадратическая ошибка:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    print(coeff_df)
    print(f'Результаты:\n{df.head(10)}')

