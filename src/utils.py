import numpy as np 
import pandas as pd
from datetime import datetime, time, timedelta
import re

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def add_keywords(data, name_column):
    stop_words = {'и', 'в', 'на', 'с', 'под', 'за',
                  'для', 'по', 'от', 'до', 'над', 
                  'через', 'у', 'о', 'об', 'при', 
                  'из', 'а', 'но', 'или', 'xx', 'хх',
                 'мм', 'ст', 'шт', 'тр', 'дл',
                 "шир", "выс", "пр"}
    
    data[name_column] = data[name_column].apply(lambda x: x.strip().lower())

    def cut_keywords(name):
        # Удаляем нетекстовые символы и числа
        name = re.sub(r'[^А-Яа-яЁё\s]', '', name)
        
        # Разбиваем строку на слова и удаляем лишние пробелы
        words = name.strip().split()
    
        # Фильтруем слова, исключая предлоги и слова, которые короче 2 символов
        keywords = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 1]
        return keywords 

    data['keywords'] = data[name_column].apply(cut_keywords)
    
    # Разворачиваем список ключевых слов
    data_keywords = data.explode('keywords')
    
    # Считаем количество каждого ключевого слова
    keywords_counts = data_keywords['keywords'].value_counts().to_dict()
    
    # Добавляем колонку с количеством вхождений каждого ключевого слова
    data_keywords['keyword_count'] = data_keywords['keywords'].map(keywords_counts)
    
    # Группируем по изначальным индексам и берем минимальное значение count для каждой строки
    keywords_count_min = data_keywords.groupby(data_keywords.index)['keyword_count'].min()
    
    # Объединяем результат с исходным DataFrame
    data = data.join(keywords_count_min.rename('keywords_count'), how='left')
    
    return data   
    
def extract_thickness(material):
    try:
        material = material.lower().replace(',', '.').split()
        thickness = None
        for word in material:
            # Проверка на наличие формата типа "60*30*2,0"
            if '*' in word:
                thickness = float(word.split('*')[-1])
                break
            # Проверка на наличие числа с точкой или без нее
            elif all(char.isdigit() or char == '.' for char in word):
                # Исключаем очевидные ошибки, например "304" или "60*30"
                if '.' in word or len(word) < 3:
                    thickness = float(word)
                    break
        if thickness is not None and thickness > 0 and thickness < 20:
            return thickness
        else:
            return 'Ошибка'
    except:
        return 'Ошибка'
    
def get_material_mark(name):
    materials_dict = {
        ('эл/св', 'труба', 'проф'): 'Сталь',
        ('нерж', 'aisi', 'нерж.'): 'Нержавейка',
        ('09г2с',): '09Г2С',
        ('д16', 'дюраль', "амг2", "амг3", "амг5"): 'Алюминий',
        ('65г',): "65Г",
        ('латунь', 'л63'): 'Латунь',
        ('оцинк',): 'Оцинковка',
        ('cor-ten',): 'CORTEN',
        ('медь',): 'медь',
        ('бронза',): 'бронза',
        ('hardox',): 'Hardox',
        ('60с2а',): '60С2А',
        ('титан',): 'Титан'
    }
    
    for key in materials_dict:
        if any(k in name for k in key):
            return (materials_dict[key])
    return 'Сталь'


def convert_to_seconds(time_str):
    # Убедимся, что time_str является строкой
    if not isinstance(time_str, str):
        try:
            time_str = str(time_str)
        except Exception as e:
            print(f"Ошибка конвертации: {e}")
            return None

    # Попытка парсинга строки времени с миллисекундами
    try:
        if '.' in time_str:
            time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
        else:
            time_obj = datetime.strptime(time_str, "%H:%M:%S")
    except ValueError as e:
        print(f"Ошибка парсинга: {e}")
        return None
    
    # Преобразование в timedelta
    delta = timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second, microseconds=time_obj.microsecond)
    
    # Получение общего количества секунд и округление до целого числа
    total_seconds = round(delta.total_seconds())
    
    return int(total_seconds)

def get_avg_group_speed(data):
    d = data.copy()
    
    # Вычисляем среднее время для каждой группы
    d['avg_group_time'] = d.groupby(['Thickness', 'Material_mark'])['Time'].transform('mean')
    
    # Вычисляем среднюю длину реза для каждой группы
    d['avg_group_length'] = d.groupby(['Thickness', 'Material_mark'])['Cut_length'].transform('mean')
    
    # Вычисляем среднюю скорость
    d['avg_group_speed'] = d['avg_group_length'] / d['avg_group_time']
    
    return d['avg_group_speed']

def sheet_size_filter(data):
    list_hor = (
        (data['Operation']=='ЛР') &
        (data['Length'] > 0) & (data['Length'] < 3000) &
        (data['Width'] > 0) & (data['Width'] < 1500)
    )
    list_vert = (
        (data['Operation']=='ЛР') &
        (data['Length'] > 0) & (data['Length'] < 1500) &
        (data['Width'] > 0) & (data['Width'] < 3000)
    )
    tube_hor = (
        (data['Operation']=='ТР') &
        (data['Length'] > 0) & (data['Length'] < 6000) &
        (data['Width'] > 0) & (data['Width'] < 1000)
    )
    tube_vert = (
        (data['Operation']=='ТР') &
        (data['Length'] > 0) & (data['Length'] < 1000) &
        (data['Width'] > 0) & (data['Width'] < 6000)
    )
    
    data = data[(list_hor | list_vert) | (tube_hor | tube_vert)]
    return data

def add_one_hot(data, name_column):
    one_hot = OneHotEncoder(dtype=int)
    encoded_array = one_hot.fit_transform(data[[name_column]]).toarray()
    temp = pd.DataFrame(encoded_array, columns=one_hot.get_feature_names_out([name_column]),
                    index=data.index)
    data = pd.concat([data, temp], axis=1)
    return data

def fill_nan(data, columns, fill_col):
    scaler = StandardScaler()
    
    # Масштабируем только выбранные столбцы
    scaled_df = pd.DataFrame(scaler.fit_transform(data[columns]), columns=columns, index=data.index)
    
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    data_input = pd.DataFrame(imputer.fit_transform(scaled_df), columns=columns, index=data.index)
    
    # Преобразуем масштабированные данные обратно к исходному масштабу
    data_input = pd.DataFrame(scaler.inverse_transform(data_input), columns=columns, index=data.index)
    
    # Заполняем пропуски в исходном DataFrame
    data[fill_col] = data_input[fill_col]
    
    
def print_grafs(data, features, n_cols):
    n_rows = len(features) // n_cols + len(features) % n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 30))
    
    for index, name in enumerate(features):
        col = index % n_cols
        row = index // n_cols
        sorted_df = data.sort_values(by=name)
        x = sorted_df[name]
        y = sorted_df['Time']
        ax[row, col].plot(x, y)
        ax[row, col].set_xlabel(name)
        if pd.api.types.is_string_dtype(data[name]) or pd.api.types.is_categorical_dtype(data[name]):
            ax[row, col].set_xticklabels(ax[row, col].get_xticklabels(), rotation=90)
            ax[row, col].set_xticklabels(ax[row, col].get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.show()
def get_and_print_metrics(y_pred, y_true):
    # Вычисление RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse}")
    # Вычисление MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    # Вычисление SMAPE
    smape_value = smape(y_true, y_pred)
    print(f"SMAPE: {smape_value:.2f}%")
    
    r2 = r2_score(y_true, y_pred)
    print(f"R^2: {r2:.2f}")
    
    absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
    percentage_within = np.mean(absolute_percentage_error <= 0.20) * 100
    print(f"Процент значений с ошибкой не более 20%: {percentage_within:.2f}%")
    
    # Вычисление WAPE
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    print(f"WAPE: {wape:.2f}%")
    
    return {'rmse':rmse,
           'mape':mape,
           'smape':smape_value,
           'R^2':r2,
           'percentage_within_20': percentage_within,
           'wape': wape}