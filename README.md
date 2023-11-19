# apartment_appraiser
A system that can estimate the cost of apartments\
Система, позволяющая оценить стоимость квартир.
Через API:
- можно загрузить данные для обучения модели;
- загрузить данные для оценки;
- выгрузить модель


# build&up image
При запуске из папки проекта (где лежит Dockerfile)\
```docker build -t apartment_appraiser . ```, где "apartment_appraiser" - имя собираемого образа\
```docker run -d --name appraiser_cont -p 80:80 apartment_appraiser```, где "appraiser_cont" - имя поднимаемого контейнера, "apartment_appraiser" - имя используемого образа

# use
API доступна по адресу ```0.0.0.0:80/docs```

# description
- Для оценки используется модель множественной линейной регрессии из scikit-learn
- Используемые параметры для оценки: city_id, district_id, street_id, price, area_total, floor_num, rooms_cnt, building_year, wall_id, type
- При подготовке данных производится очистка от статистических выбросов (оценка с помощью qq-plot), обрабатываются NaN (в зависимости от поля либо заполняются модой, либо удаляются)
- Оценка используемых полей производится, по коеффициентам, формируемым моделью (приходят в ответе на "create_model")
- В файле для Jupyter можно посмотреть графики и проверки
