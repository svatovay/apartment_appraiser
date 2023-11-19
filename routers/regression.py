import datetime
import os
import tempfile

from fastapi import APIRouter, UploadFile
from starlette.responses import FileResponse

from sklearn.preprocessing import LabelEncoder

from services import data_handlers as handler
from services import algorithm as alg

router = APIRouter(
    prefix='/regression',
    tags=['LinearRegression'],
)


@router.post('/create_model')
async def create_linear_regression(training_dataset: UploadFile):
    _cols_to_wash = ['sold_price', 'price', 'floor_num', 'rooms_cnt', 'building_year', 'area_total']
    _cols_to_label = ['floor_num', 'rooms_cnt', 'wall_id', 'type']
    _cols_to_fill = [*_cols_to_label, 'building_year']
    _cols_to_independent = ['city_id', 'district_id', 'street_id', 'price', 'area_total',
                            'floor_num', 'rooms_cnt', 'building_year', 'wall_id',
                            'type']

    # Создание df
    u_file = await training_dataset.read()
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(u_file)
        dataset = handler.read_data(fp.name, date_cols=['date_sold'], date_format='%Y-%m-%d')

    # Чистка от NaN
    dataset = handler.drop_na(dataset, ['sold_price', 'price', 'area_total'])

    # Чистка от выбросов
    for _col in _cols_to_wash:
        dataset = handler.wash_invalid_values(dataset, _col)
    else:
        del _col, _cols_to_wash

    # Разметка
    le = LabelEncoder()
    for _col in _cols_to_label:
        dataset = handler.label_values(dataset, _col, le)
    else:
        del _col, _cols_to_label, le

    # Заполнение Nan
    for _col in _cols_to_fill:
        dataset = handler.fill_na(dataset, _col, 'mode')
    else:
        del _col, _cols_to_fill

    x, y = handler.create_samples(dataset, _cols_to_independent, 'sold_price')
    lr_model, x_test, y_test = alg.create_model({'x': x, 'y': y})
    is_checked = alg.check_model(lr_model, x_test, y_test)
    if is_checked:
        model_name = f'lr_{datetime.datetime.now().isoformat()}'
        model_coeffs = alg.get_coefficients(lr_model, x.columns).to_dict()
        alg.save_model(lr_model, model_name)
        return {'model_name': model_name, 'model_coefficients': model_coeffs}
    else:
        return 'Not Success'


@router.post('/predict')
async def create_predict(to_predict: UploadFile):
    _cols_to_wash = ['price', 'floor_num', 'rooms_cnt', 'building_year', 'area_total']
    _cols_to_label = ['floor_num', 'rooms_cnt', 'wall_id', 'type']
    _cols_to_fill = [*_cols_to_label, 'building_year']
    _cols_to_independent = ['city_id', 'district_id', 'street_id', 'price', 'area_total',
                            'floor_num', 'rooms_cnt', 'building_year', 'wall_id',
                            'type']
    u_file = await to_predict.read()
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(u_file)
        dataset = handler.read_data(fp.name, date_cols=['date_sold'], date_format='%Y-%m-%d')

        # Чистка от NaN
        dataset = handler.drop_na(dataset, ['price', 'area_total'])

        # Чистка от выбросов
        for _col in _cols_to_wash:
            dataset = handler.wash_invalid_values(dataset, _col)
        else:
            del _col, _cols_to_wash

        # Разметка
        le = LabelEncoder()
        for _col in _cols_to_label:
            dataset = handler.label_values(dataset, _col, le)
        else:
            del _col, _cols_to_label, le

        # Заполнение Nan
        for _col in _cols_to_fill:
            dataset = handler.fill_na(dataset, _col, 'mode')
        else:
            del _col, _cols_to_fill

    lr_model = alg.load_model()
    predict = alg.get_predict(lr_model, dataset[_cols_to_independent])
    dataset['sold_price'] = predict

    with tempfile.NamedTemporaryFile(delete=False) as fp:
        dataset.to_csv(fp.name)
    file_name = f'predicted_{datetime.datetime.now().isoformat()}.csv'
    return FileResponse(fp.name, media_type="application/octet-stream", filename=file_name)


@router.get('/model')
def get_model(model_name: str | None = None):
    _path = f'lr_models/{model_name}'
    _models = os.listdir('lr_models')
    if model_name in _models:
        _path = f'lr_models/{model_name}'
        return FileResponse(_path, media_type="application/octet-stream", filename=model_name)
    else:
        return 'No such file'

