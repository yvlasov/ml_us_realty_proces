# «Модель прогнозирования стоимости жилья для агентства недвижимости»

> Этот учебный проект является типовой задачей, с которой может встретиться в работе дата-сайентист.
> Код приложения и доп информацию можно найти в github репозитории [yvlasov/ml_us_realty_proces](https://github.com/yvlasov/ml_us_realty_proces).

> Работа выполнена [Власовым Юрием](http://t.me/yuvlasov)

## Задача следующая:
 Aгентство недвижимости столкнулось с проблемой — риелторы тратят слишком много времени на сортировку объявлений и поиск выгодных предложений. Поэтому скорость их реакции и качество анализа не дотягивают до уровня конкурентов. Это сказывается на финансовых показателях агентства.

## Ваша цель:
Разработать модель машинного обучения, которая поможет обрабатывать объявления и увеличит число сделок и прибыль агентства.

## Выполнение задачи

> Все подробности о ходе выполнения и подробностях анализа можно найти в [IPythonNotebook](https://drive.google.com/file/d/1q1anIuwzIdnz6n4tvnnocLgaEj1b00Sl/view?usp=sharing)

## Краткое описание работы

* Для предсказания цены недвижимости в США был использован [предложенный набор данных](https://yvlasov-share.s3.amazonaws.com/diploma_project_data.csv.zip).
* Набор данных содержал несколько сложных вложенных полей с информацией о объекте и образовательных учреждениях в окрестности.
* Изучив имеющиеся аналитические данные стало очевидно что важным критерием в США являются именно образовательные учреждения в окрестности объекта. Мной было принято решение о досканальной проработке этих вложенных данных что при дальнейшем анализе ```FeatureImportance``` показало положительный результат.
* Также для данной задачи было принято решение использовать как интуитивно понятную метрику сравнения **Mean Absolute Percentage Error (MAPE)**.
* В качестве контрольной модели был использован ```KNeighborsRegressor(n_neighbors=10)```. Контрольный эксперимент показал **MAPE: 61.3%**
* Как основная гепотеза была выбрана модель **CatBoostRegressor** эксперимент показал **MAPE: 0.28%**

> PS: Был выполнен эксперимент с SGDRegression - но ввиду дополнительных сложностей по нормализации данных было решено его не использовать и значимого прироста относительно KNN он не показал при первом эксперименте.

### LOG обучения:
Параметры CatBoost выбранные после нескольких попыток обучения: 
```
iterations=10000
depth=10
learning_rate=0.1
loss_function='RMSE'
eval_metric='MAPE'
```
LOG обучения:
```
...
6000:	learn: 0.1477715	test: 0.2766938	best: 0.2766897 (5997)	total: 37m 5s	remaining: 24m 43s
6025:	learn: 0.1474407	test: 0.2765891	best: 0.2765749 (6023)	total: 37m 15s	remaining: 24m 34s
6050:	learn: 0.1471457	test: 0.2765359	best: 0.2765216 (6042)	total: 37m 23s	remaining: 24m 23s
6075:	learn: 0.1469125	test: 0.2765509	best: 0.2765216 (6042)	total: 37m 32s	remaining: 24m 14s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.2765215546
bestIteration = 6042
```

## Анализ результата:
> Графики на основании которых делался вывод можно найти вконце [IPythonNotebook](https://drive.google.com/file/d/1q1anIuwzIdnz6n4tvnnocLgaEj1b00Sl/view?usp=sharing)

1. Очевидно что наибольшее значение оказывает площадь объекта и год постройки.
2. Большое значение оказывает количество сан-узлов
3. Как и ожидалось исходя из анализа метаданных задачи важную роль играет информация о ближайших учебных заведениях.
4. Весь набор данных о рейтинге расстоянии и наличию различных классов образования, в окрестности обьекта показал высокую значимость.
4. Важным фактором оказался и тип  объекта.
5. Так же факт расположения обьекта в штатах CA, NY, WA, TX, FL является значимым.

# ВЫВОД:
Результат хорошо интерпретируется и соответствует интуитивному пониманию и жизненному опыту. Так же имеющиеся в публичном доступе знания по аналогичным задачам подтверждают что информация о образовательных учреждениях сильно кореллирует с ценой недвижимости на рынке США.
Модель KNN показала MAPE около 52% в то время как CatBost 27.19%

# Эксплуатация

## Сборка Docker

### Запустите сборку и обучение:
```
docker build . \
    --progress=plain \
    -t us_realty_predict
```
После сборку у вас появится образ Docker ```us_realty_predict```, далее вы можете его запустить как API сервер.
>ОШИБКА: При возникновении ошибки при сборке попробуйте перезапустить с параметром ```--no-cache```.

### Запуск сервера и выполнение запросов

#### Запуск сервера собранного выше Docker образа ```us_realty_predict```:
```
 docker run -it \
    --name us_realty_predict \
    --rm \
    -p 8080:8080 \
    -e FLASK_DEBUG=1 \
    us_realty_predict
```

#### Пример запроса информации о моделях:
```
curl http://127.0.0.1:8080/model/info
```
Ответ:
```
US relty price prediction v0.99.0 MAPE:27.96
```

#### Пример запроса информации о фичах:
```
curl http://127.0.0.1:8080/model/features           
```
Ответ:
```
["baths", "fireplace", "sqft", "beds", "stories", "hf_year_built", "hf_lotsize", "hf_remodeled_year", "hf_parking", "priv_pool", "schl_rating_mean", "schl_distnce_mean", "schl_gr_p_top_rate_dist", "schl_gr_p_top_rate", "schl_gr_p_min_dist", "schl_gr_pk_top_rate_dist", "schl_gr_pk_top_rate", "schl_gr_pk_min_dist", "schl_gr_k_top_rate_dist", "schl_gr_k_top_rate", "schl_gr_k_min_dist", "schl_gr_1_top_rate_dist", "schl_gr_1_top_rate", "schl_gr_1_min_dist", "schl_gr_2_top_rate_dist", "schl_gr_2_top_rate", "schl_gr_2_min_dist", "schl_gr_3_top_rate_dist", "schl_gr_3_top_rate", "schl_gr_3_min_dist", "schl_gr_4_top_rate_dist", "schl_gr_4_top_rate", "schl_gr_4_min_dist", "schl_gr_5_top_rate_dist", "schl_gr_5_top_rate", "schl_gr_5_min_dist", "schl_gr_6_top_rate_dist", "schl_gr_6_top_rate", "schl_gr_6_min_dist", "schl_gr_7_top_rate_dist", "schl_gr_7_top_rate", "schl_gr_7_min_dist", "schl_gr_8_top_rate_dist", "schl_gr_8_top_rate", "schl_gr_8_min_dist", "schl_gr_9_top_rate_dist", "schl_gr_9_top_rate", "schl_gr_9_min_dist", "schl_gr_10_top_rate_dist", "schl_gr_10_top_rate", "schl_gr_10_min_dist", "schl_gr_11_top_rate_dist", "schl_gr_11_top_rate", "schl_gr_11_min_dist", "schl_gr_12_top_rate_dist", "schl_gr_12_top_rate", "schl_gr_12_min_dist", "state_AL", "state_AZ", "state_CA", "state_CO", "state_DC", "state_DE", "state_FL", "state_GA", "state_IA", "state_IL", "state_IN", "state_KY", "state_MA", "state_MD", "state_ME", "state_MI", "state_MO", "state_MS", "state_MT", "state_NC", "state_NJ", "state_NV", "state_NY", "state_OH", "state_OK", "state_OR", "state_PA", "state_SC", "state_TN", "state_TX", "state_UT", "state_VA", "state_VT", "state_WA", "state_WI", "type_condo", "type_coop", "type_high_rise", "type_land", "type_mobile_manufactured", "type_multi_family", "type_ranch", "type_single_family", "type_townhouse", "type_traditional", "status_active", "status_contingent", "status_for_sale", "status_foreclosed", "status_foreclosure", "status_nan", "status_new", "status_pending", "status_pre_foreclosure", "status_under_contract", "hf_cooling_central", "hf_cooling_cooling_system", "hf_cooling_electric", "hf_cooling_gas", "hf_cooling_has_cooling", "hf_cooling_heat_pump ", "hf_cooling_nan", "hf_cooling_no_data", "hf_cooling_other", "hf_cooling_wall", "hf_heating_air", "hf_heating_baseboard", "hf_heating_central", "hf_heating_electric", "hf_heating_gas", "hf_heating_heat_pump ", "hf_heating_nan", "hf_heating_no_data", "hf_heating_other", "hf_heating_wall"]
```

#### Пример запроса предсказания для модели:
```
curl -X POST http://127.0.0.1:8080/model/predict \
-H "Content-Type: application/json" \
-d '{
   "model_version":"0.99.0",
   "features":{
      "bath":"1.0",
       ...
    }
    }'
```
Ответ если вы не указали все фичи:
```
"Missing features: ['baths', 'fireplace', 'sqft', 'beds', 'stories', ...]'
```
