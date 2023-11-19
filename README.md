# apartment_appraiser
A system that can estimate the cost of apartments
Система, позволяющая оценить стоимость квартир.
Через API:
- можно загрузить данные для обучения модели;
- загрузить данные для оценки;
- выгрузить модель


# build&up image
При запуске из папки проекта (где лежит Dockerfile)
```docker build -t apartment_appraiser . ```, где "apartment_appraiser" - имя собираемого образа
```docker run -d --name appraiser_cont -p 80:80 apartment_appraiser```, где "appraiser_cont" - имя поднимаемого контейнера, "apartment_appraiser" - имя используемого образа

# use
Api доступна по адресу ```0.0.0.0:80/docs```
