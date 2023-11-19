import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    # Чтение
    dataset = pd.read_csv('sold_flats_2020-09-30.csv', parse_dates=['date_sold'], date_format='%Y-%m-%d')
    le = LabelEncoder()
    regressor = LinearRegression()

    # Нормализация
    dataset = dataset.dropna(subset=['sold_price', 'price', 'area_total'])
    # Границы для цен продаж
    Q3_sold_price = dataset['sold_price'].quantile(0.75)
    Q1_sold_price = dataset['sold_price'].quantile(0.25)
    IQR_sold_price = Q3_sold_price - Q1_sold_price

    # Границы для цен
    Q3_price = dataset['price'].quantile(0.75)
    Q1_price = dataset['price'].quantile(0.25)
    IQR_price = Q3_price - Q1_price

    # Границы для этажа
    Q3_floor_num = dataset['floor_num'].quantile(0.75)
    Q1_floor_num = dataset['floor_num'].quantile(0.25)
    IQR_floor_num = Q3_floor_num - Q1_floor_num

    # Границы для кол-ва комнат
    Q3_rooms_cnt = dataset['rooms_cnt'].quantile(0.75)
    Q1_rooms_cnt = dataset['rooms_cnt'].quantile(0.25)
    IQR_rooms_cnt = Q3_rooms_cnt - Q1_rooms_cnt

    # Границы для года постройки
    Q3_building_year = dataset['building_year'].quantile(0.75)
    Q1_building_year = dataset['building_year'].quantile(0.25)
    IQR_building_year = Q3_building_year - Q1_building_year

    # Границы для общей площади
    Q3_area_total = dataset['area_total'].quantile(0.75)
    Q1_area_total = dataset['area_total'].quantile(0.25)
    IQR_area_total = Q3_area_total - Q1_area_total

    # Очистка выбросов
    dataset = dataset[(dataset['sold_price'] > (Q1_sold_price - 1.5 * IQR_sold_price)) & (
            dataset['sold_price'] < (Q3_sold_price + 1.5 * IQR_sold_price))]
    dataset = dataset[
        (dataset['price'] > (Q1_price - 1.5 * IQR_price)) & (dataset['price'] < (Q3_price + 1.5 * IQR_price))]
    dataset = dataset[(dataset['floor_num'] > (Q1_floor_num - 1.5 * IQR_floor_num)) & (
            dataset['floor_num'] < (Q3_floor_num + 1.5 * IQR_floor_num))]
    dataset = dataset[(dataset['rooms_cnt'] > (Q1_rooms_cnt - 1.5 * IQR_rooms_cnt)) & (
            dataset['rooms_cnt'] < (Q3_rooms_cnt + 1.5 * IQR_rooms_cnt))]
    dataset = dataset[(dataset['building_year'] > (Q1_building_year - 1.5 * IQR_building_year)) & (
            dataset['building_year'] < (Q3_building_year + 1.5 * IQR_building_year))]
    dataset = dataset[(dataset['area_total'] > (Q1_area_total - 1.5 * IQR_area_total)) & (
            dataset['area_total'] < (Q3_area_total + 1.5 * IQR_area_total))]

    le.fit(dataset['floor_num'])
    dataset['floor_num'] = le.transform(dataset['floor_num'])
    dataset['floor_num'] = dataset['floor_num'].fillna(dataset['floor_num'].mode()[0])

    le.fit(dataset['rooms_cnt'])
    dataset['rooms_cnt'] = le.transform(dataset['rooms_cnt'])
    dataset['rooms_cnt'] = dataset['rooms_cnt'].fillna(dataset['rooms_cnt'].mode()[0])

    dataset['building_year'] = dataset['building_year'].fillna(dataset['building_year'].mode()[0])

    le.fit(dataset['wall_id'])
    dataset['wall_id'] = le.transform(dataset['wall_id'])
    dataset['wall_id'] = dataset['wall_id'].fillna(dataset['wall_id'].mode()[0])

    le.fit(dataset['type'])
    dataset['type'] = le.transform(dataset['type'])
    dataset['type'] = dataset['type'].fillna(dataset['type'].mode()[0])

    # Обучение
    X = dataset[
        ['city_id', 'district_id', 'street_id', 'price', 'area_total',
         'floor_num', 'rooms_cnt', 'building_year', 'wall_id',
         'type']
    ]
    Y = dataset['sold_price']

    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=0)
    regressor.fit(X_train, Y_train)

    # Проверка
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Коэффициент'])
    Y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Фактическая': Y_test, 'Предсказанная': Y_pred})

    print('Средняя абсолютная ошибка:', metrics.mean_absolute_error(Y_test, Y_pred))
    print('Средняя квадратическая ошибка:', metrics.mean_squared_error(Y_test, Y_pred))
    print('Среднеквадратичная ошибка:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    print(df['Фактическая'].mean())
    print(df['Предсказанная'].mean())
    print(coeff_df)
    print(f'Результаты:\n{df.head(10)}')
