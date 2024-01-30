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

Для предсказания цены недвижимости в США был использован [предложенный набор данных](https://yvlasov-share.s3.amazonaws.com/diploma_project_data.csv.zip).
Набор данных содержал несколько сложных вложенных полей с информацией о объекте и образовательных учреждениях в окрестности.
Изучив имеющиеся аналитические данные стало очевидно что важным критерием в США являются именно образовательные учреждения в окрестности объекта. Мной было принято решение о досканальной проработке этих вложенных данных что при дальнейшем анализе ```FeatureImportance``` показало положительный результат.
Также для данной задачи было принято решение использовать как интуитивно понятную метрику сравнения **Mean Absolute Percentage Error (MAPE)**.
В качестве контрольной модели был использован ```KNeighborsRegressor(n_neighbors=10)```. Контрольный эксперимент показал **MAPE: ~52%**
> Был выполнен эксперимент с SGDRegression - но ввиду дополнительных сложностей по нормализации данных было решено его не использовать и значимого прироста относительно KNN он не показал при первом эксперименте.
Как основная гепотеза была выбрана модель **CatBoostRegressor**.

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