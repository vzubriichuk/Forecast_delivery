#!/usr/bin/env python
# coding: utf-8

print('importing libraries -- start')
import pandas as pd
from datetime import datetime, date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sklearn.metrics as metrics
import urllib
from sqlalchemy import create_engine
import pyodbc
import scipy
from os import getcwd, path
import sys
import time

def writelog(e):
    """ Write error log into file log.txt.
    """
    fname = path.join(getcwd(), 'log.txt')
    now = time.localtime()

    with open(fname, 'a') as f:
        f.write('{} {}\n'.format(time.strftime("%d-%m-%Y %H:%M:%S", now), e))


print('importing libraries -- done')

print('establishing sql connection 1/2 -- start')
# задаем подключение к sql server к таблице, в которой лежат данные о заказах
server_import_orders = 's-kv-center-s64'
database_import_orders = 'PromoPricing'

params_import_orders = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};"
                                               f"SERVER= {server_import_orders};"
                                               f"DATABASE={database_import_orders};"
                                               "Trusted_Connection=yes")

engine_import_orders = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params_import_orders).connect()

# отбираем данные с 21.09.2020. чистим их от отмененных заказов

sql_query_orders = "SELECT [orderNumber], operReadyDate  ,[filialId]      ,[deliveryServiceTypeId]  FROM [PromoPricing].[dbo].[Orders] ord with(nolock)  where 1=1    and operReadyDate >= '20200921'    and cancelDate is null "

# загружаем таблицу с заказами
orders = pd.read_sql(sql=sql_query_orders, con=engine_import_orders)
print('establishing sql connection 1/2 -- done')

print('establishing sql connection 2/2 -- start')
# задаем подключение к sql server к таблице, в которой лежат данные о связке филиал-город
server_import_cities = 'S-KV-CENTER-S31'
database_import_cities = 'Transport_Analyst'

params_import_cities = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};"
                                               f"SERVER= {server_import_cities};"
                                               f"DATABASE={database_import_cities};"
                                               "Trusted_Connection=yes")

engine_import_cities = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params_import_cities).connect()

sql_query_cities = "SELECT        filid      ,CityName  FROM Transport_Analyst.dbo.Filials_City_GEO_info  with(nolock)"

print('establishing sql connection 2/2 -- done')

print('data preprocessing 1/4 -- start')
# выгружаем таблицу со связкой филиал-город
cities = pd.read_sql(sql=sql_query_cities, con=engine_import_cities)

# обьединяем таблицу с заказами с таблицей с городами.
imported_data = orders.merge(cities, how='left', left_on='filialId', right_on='filid').drop(columns='filid')

# колонку operReadyDate округлим вниз до часа
dataset = imported_data.copy()
dataset['operReadyDate'] = dataset['operReadyDate'].dt.floor('h')

# 1. создаем календарь истории доставок, начиная с первой существующей в dataset даты и заканчивая последней
# 2. для корректного урезания данных по дате для обучения модели введем cut_date - дату,
#    которая является минимальной из вариантов ("вчера", "дата последней доставки")

start_date = datetime.fromtimestamp(np.sort(dataset['operReadyDate'].unique())[0].item() / 10 ** 9).replace(hour=0,
                                                                                                            minute=0,
                                                                                                            second=0,
                                                                                                            microsecond=0)
end_date = np.sort(dataset['operReadyDate'].unique())[-1]
cut_date = min(datetime.fromtimestamp(np.sort(dataset['operReadyDate'].unique())[-1].item() / 10 ** 9),
               datetime.now() - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)

# для заполнения нулевых значений доставок создадим сначала таблицу, соединив календарь дат, филиалы и доставки
dates = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='h'), columns=['operReadyDate'])
fils = dataset[['filialId', 'CityName']].copy().drop_duplicates()
deliv = dataset[['deliveryServiceTypeId']].copy().drop_duplicates()

fils['key'] = 0
deliv['key'] = 0
dates['key'] = 0

alldates = fils.merge(deliv, how='outer', on='key').merge(dates, how='outer', on='key').drop(columns='key')

dataset = dataset.merge(alldates, how='outer', on=['operReadyDate', 'filialId',
                                                   'deliveryServiceTypeId',
                                                   'CityName'])

# высчитаем количество доставок в каждый час на каждом филиале и для каждого типа доставки
dataset = dataset.groupby(['filialId', 'operReadyDate',
                           'deliveryServiceTypeId',
                           'CityName'], as_index=False).count()

dataset['year'] = dataset['operReadyDate'].dt.isocalendar().year
dataset['week'] = dataset['operReadyDate'].dt.isocalendar().week
dataset['weekday'] = dataset['operReadyDate'].dt.isocalendar().day
dataset['hour'] = dataset['operReadyDate'].dt.hour

# усекаем лишние данные (например, плановые доставки на "завтра", доставки "сегодня" и т.д при помощи cut_date)
grouped = dataset[dataset['operReadyDate'] <= cut_date].copy().drop('operReadyDate', 1)

# создаем глобальный календаRF training рь недель для преобразования номера недели и года в week_index - возрастающий индекс недели
end_date_calendar = datetime.now() + timedelta(days=40)

calendar = pd.DataFrame(pd.date_range(start=start_date, end=end_date_calendar, freq='w'), columns=['operReadyDate'])
calendar['year'] = calendar['operReadyDate'].dt.isocalendar().year
calendar['week'] = calendar['operReadyDate'].dt.isocalendar().week

week_index = calendar[['week', 'year']].drop_duplicates().sort_values(by=['year',
                                                                          'week']).reset_index(
    drop=True).reset_index().rename(columns={'index': 'week_index'})
week_index.loc[:, ['week_index']] += 1

# добавляем к данным о доставках колонку с week_index
grouped = grouped.merge(week_index, how='left', on=['week', 'year'])

# разбиваем данные на X_train - независимые переменные, Y_train - зависимая переменная
X_train = grouped.drop(columns=['orderNumber', 'CityName', 'week', 'year'])
Y_train = grouped['orderNumber']

print('data preprocessing 1/4 -- done')

print('RF training -- start')

# обучаем случайный лес
rf = RandomForestRegressor(max_features='auto', n_jobs=-1, random_state=5, n_estimators=600,
                           bootstrap=True, max_samples=0.8, min_samples_split=2, oob_score=True, verbose=1)
rf_fit = rf.fit(X_train, Y_train)

print('RF training -- done')
print('data processing 2/4 -- start')

# создаем календарь дат от сегодня и на 32 дня вперед по часам
month_start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
month_end_date = month_start_date + timedelta(days=31, hours=23)

# days7 - дата, до которой мы будем добавлять данные из прогноза в тренд
# days11 - дата, до которой мы будем добавлять данные в почасовой прогноз на 11 дней
days7 = month_start_date + timedelta(days=6, hours=23)
days11 = month_start_date + timedelta(days=10, hours=23)

month_dates = pd.DataFrame(pd.date_range(start=month_start_date, end=month_end_date, freq='h'),
                           columns=['operReadyDate'])
month_dates['key'] = 0

# обьединяем таблицы дат, филиалов и типов доставки и week_index
to_pred_month = fils.merge(deliv, how='outer', on='key').merge(month_dates, how='outer', on='key').drop('key', 1)
to_pred_month['year'] = to_pred_month['operReadyDate'].dt.isocalendar().year
to_pred_month['week'] = to_pred_month['operReadyDate'].dt.isocalendar().week
to_pred_month['weekday'] = to_pred_month['operReadyDate'].dt.isocalendar().day
to_pred_month['hour'] = to_pred_month['operReadyDate'].dt.hour
to_pred_month = to_pred_month.merge(week_index, how='left', on=['week', 'year'])

to_pred_month.drop(columns=['operReadyDate', 'CityName', 'week', 'year'])

print('data processing 2/4 -- done')

print('RF predicting -- start')
# строим прогноз и округляем его по математическим правилам
predicted_rf = rf_fit.predict(to_pred_month.drop(columns=
                                                 ['operReadyDate', 'CityName', 'week', 'year'],
                                                 inplace=False))
predicted_rf_rounded = predicted_rf.round().astype(int)

print('RF predict -- done')
print('data processing 3/4 -- start')

# создаем табличку с прогнозом и остальными данными (филиал, тип доставки и т.д) на 32 дня по часам
month = pd.concat([to_pred_month.reset_index(drop=True, inplace=False),
                   pd.Series(predicted_rf_rounded, name='forecast')], axis=1)

# добавляем predicted month (первую неделю) к входным данным и высчитываем тренд на основе количества доставок в городе за неделю

# добавляем из спрогнозированного month только первую неделю к входным данным и высчитываем тренд на основе количества доставок в городе за неделю
month.rename(columns={'forecast': 'orderNumber'}, inplace=True)
grouped = pd.concat([grouped, month[month['operReadyDate'] <= days7].drop(columns='operReadyDate', inplace=False)],
                    axis=0)

# группируем по week/city/delivery

grouped_weekcity = grouped.drop(columns=['year', 'week', 'hour', 'filialId'],
                                inplace=False).groupby(by=
                                                       ['deliveryServiceTypeId', 'week_index', 'CityName'],
                                                       as_index=False).aggregate(
    {'orderNumber': 'sum', 'weekday': 'nunique'})

# считаем количество дней в каждой неделе и отсекаем неполные недели из построения тренда
grouped_weekcity = grouped_weekcity[grouped_weekcity.weekday == 7].drop(columns='weekday')
print('data processing 3/4 -- done')

print('LR training -- start')

# строим линейную регрессию для выявления тренда доставок по городу от недели к неделе
# записываем полученные коэффициенты уравнения в табличку trends
lr = LinearRegression()
cities = grouped_weekcity.CityName.unique()
city_list = []
delivery_list = []
coef_list = []
intercept_list = []
for city in cities:
    for delivery_type in deliv.deliveryServiceTypeId.unique():
        X_train_lr = grouped_weekcity[(grouped_weekcity.CityName == city) &
                                      (
                                                  grouped_weekcity.deliveryServiceTypeId == delivery_type)].week_index.values.reshape(
            -1, 1)
        Y_train_lr = grouped_weekcity[(grouped_weekcity.CityName == city) &
                                      (grouped_weekcity.deliveryServiceTypeId == delivery_type)].orderNumber
        fit = lr.fit(X_train_lr, Y_train_lr)
        city_list.append(city)
        delivery_list.append(delivery_type)
        coef_list.append(float(fit.coef_))
        intercept_list.append(float(fit.intercept_))


print('LR training -- done')

print('data processing 4/4 -- start')
trends = pd.DataFrame(data={'CityName': city_list,
                            'deliveryServiceTypeId': delivery_list,
                            'coef': coef_list,
                            'intercept': intercept_list})

# прогнозируем количество продаж для недель из month по городу, неделе, типу доставки
weeks_to_predict = pd.DataFrame(list(range(month.week_index.min(), month.week_index.max() + 1)),
                                columns=['week_index'])
weeks_to_predict['key'] = 0
trends['key'] = 0

trends_weekly = trends.merge(weeks_to_predict, how='outer', on='key').drop(columns='key')
trends_weekly['forecast'] = trends_weekly['coef'] * trends_weekly['week_index'] + trends_weekly['intercept']

# на основе полученного прогноза считаем процентный прирост доставок от недели к неделе
trends_weekly['pct'] = trends_weekly.sort_values('week_index').groupby(['CityName',
                                                                        'deliveryServiceTypeId']).forecast.pct_change() + 1
# считаем кумулятивное произведение процентного прироста доставок с увеличением номера недели(week_index)
# получаем коэффициент роста
trends_weekly['pct_cumprod'] = trends_weekly.sort_values('week_index').groupby(['CityName',
                                                                                'deliveryServiceTypeId']).pct.cumprod()
# заполняем пустые значения нулями
trends_weekly.fillna(0, inplace=True)

# обьединяем почасовой прогноз на месяц с таблицей трендов
month_trend = month.merge(trends_weekly[['CityName', 'deliveryServiceTypeId',
                                         'week_index', 'pct_cumprod']],
                          how='left', on=['CityName', 'deliveryServiceTypeId', 'week_index'])

# для первых 11 дней прогноза коэффициент роста заменяем на 1, чтобы в будущем в почасовом отчете и подневном отчете
# сумма доставок по первым 11 дням совпадала
month_trend.loc[month_trend['operReadyDate'] <= days11, 'pct_cumprod'] = 1

# корректируем первоначальный прогноз при помощи коэффициента роста
month_trend['trend_forecast'] = month_trend['orderNumber'] * month_trend['pct_cumprod']

# отбираем колонки, которые будем записывать в таблицу в БД
# для почасового прогноза сразу возьмем колонку hour
select_columns_hourly = ['filialId', 'deliveryServiceTypeId', 'operReadyDate', 'trend_forecast', 'hour']
select_columns_daily = ['filialId', 'deliveryServiceTypeId', 'operReadyDate', 'trend_forecast']

# создаем датафрейм с почасовым прогнозом на 11 дней, который будем записывать в таблицу в БД
# добавляем колонку detailLevel, которая будет указывать на то, что это почасовой прогноз
# при этом колонку operReadyDate переводим в формат даты без времени
hourly_forecast = month_trend[month_trend['operReadyDate'] <= days11].loc[:, select_columns_hourly].copy().rename(
    columns={'trend_forecast': 'forecast'})
hourly_forecast.iloc[:, :]['detailLevel'] = 'hourly'
hourly_forecast['operReadyDate'] = hourly_forecast['operReadyDate'].dt.date

# для дневного отчета на 32 дня сгруппируем прогнозы до уровня дня, после чего произведем округление прогнозов
month_trend_grouped = month_trend.drop(columns=
                                       ['hour', 'operReadyDate', 'week_index']).groupby(by=
                                                                                        ['filialId',
                                                                                         'deliveryServiceTypeId',
                                                                                         'CityName', 'year', 'week',
                                                                                         'weekday'],
                                                                                        as_index=False).sum()
month_trend_grouped['trend_forecast'] = month_trend_grouped['trend_forecast'].round(decimals=0).astype(int)

# вернем в таблицу колонку operReadyDate, восстановив дату из года, номера недели и номера дня недели
month_trend_grouped['operReadyDate'] = month_trend_grouped.year.astype(str) + '-' + month_trend_grouped.week.astype(
    str) + '-' + month_trend_grouped.weekday.astype(str)
month_trend_grouped['operReadyDate'] = pd.to_datetime(month_trend_grouped['operReadyDate'], format='%G-%V-%u').dt.date

# 1.создаем датафрейм с подневным прогнозом, который будем записывать в таблицу в БД,
# урезая при этом первые 11 дней, которые уже есть в hourly_forecast,
# 2. добавляем колонку hour с пустыми ячейками, а также параметр детализации прогноза detailLevel
daily_forecast = month_trend_grouped[month_trend_grouped['operReadyDate'] > days11.date()].loc[:,
                 select_columns_daily].copy().rename(columns={'trend_forecast': 'forecast'})
daily_forecast.iloc[:, :]['hour'] = np.nan
daily_forecast.iloc[:, :]['detailLevel'] = 'daily'

# 1.создаем переменную modified_date, которая в финальной таблице прогнозов будет служить индикатором,
# когда был сделан прогноз
# 2. соединяем 11 дней почасового прогноза и 20 дней подневного в одну таблицу с доп.параметрами detailLevel и modifiedDate
modified_date = date.today()
union_forecast = pd.concat([hourly_forecast, daily_forecast], axis=0)
union_forecast.iloc[:, :]['modifiedDate'] = modified_date

# переупорядочиваем столбцы для записи в БД
columns_order = ['filialId', 'deliveryServiceTypeId', 'detailLevel', 'operReadyDate', 'hour', 'forecast',
                 'modifiedDate']
union_forecast = union_forecast[columns_order]

print('data processing 4/4 -- done')


print('establishing sql connection (export) -- start')
# задаем подключение к sql server в базу, в которую будем записывать прогноз
server_export = 'KVCEN15-SQLS005\HEAVY005'
database_export = 'ForecastAnalysis'

params_export = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};"
                                        f"SERVER= {server_export};"
                                        f"DATABASE={database_export};"
                                        "Trusted_Connection=yes")

engine_export = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params_export, fast_executemany=True)



union_forecast_table_name = 'Ecom_Forecast_try'

# непосредственно записываем почасовой и подневной прогнозы в таблицы
# параметр if_exists должен быть append, если мы хотим сохранять историю предыдущих прогнозов

union_forecast.to_sql(name=union_forecast_table_name, con=engine_export, if_exists='append', index=False,
                          schema='dbo')


print('establishing sql connection (export) -- done')




