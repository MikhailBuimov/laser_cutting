import sys
sys.path.append('../')
from src.utils import *

def get_preprocessed_data():
    #Загружаем данные
    link = "../data/laser_cutting.csv"
    data = pd.read_csv(link, on_bad_lines="skip")
    data.rename(columns={'Наименование': 'Part_name',
                        'Материал': 'Material',
                        'Время': 'Time',
                        'Ширина': 'Width',
                        'Длина': 'Length',
                        'Длина реза': 'Cut_length',
                        'Кол-во резов': 'Cut_quantity',
                        'Операция': 'Operation'}, inplace=True)

    #Удаляем дубли
    data.drop_duplicates(subset='hash', inplace=True)


    #Добавляем ключевые слова из partname
    data = add_keywords(data, 'Part_name')

    #Парсим марку материала
    data['Material']=data['Material'].apply(lambda x: x.strip().lower())
    data['Material_mark'] = data['Material'].apply(get_material_mark)

    #Парсим толщину материала
    data['Thickness'] = data['Material'].apply(extract_thickness)
    data = data[data['Thickness']!='Ошибка']

    #Переводим время в нужный формат
    data['Time']=data['Time'].apply(convert_to_seconds)
    data=data[data['Time']!=0]

    #Добавим скорость резки
    data['avg_group_speed'] = get_avg_group_speed(data)
    
    data = sheet_size_filter(data)

    #Кодируем материал
    data = add_one_hot(data, 'Material_mark')

    
    columns=['Width', 'Length', 'Cut_length', 'Cut_quantity',
         'keywords_count', 'Thickness', 'Time',
         'avg_group_speed','Material_mark_09Г2С', 'Material_mark_60С2А', 'Material_mark_65Г',
         'Material_mark_CORTEN', 'Material_mark_Hardox', 'Material_mark_Алюминий', 
         'Material_mark_Латунь', 'Material_mark_Нержавейка', 'Material_mark_Оцинковка',
         'Material_mark_Сталь', 'Material_mark_Титан', 'Material_mark_бронза',
         'Material_mark_медь']  
    
    #Заполняем пропуски
    fill_nan(data, columns, 'keywords_count')
    
    features = ['Width', 'Length', 'Cut_length', 'Cut_quantity',
            'keywords_count', 'Material_mark', 'Thickness', 
            'avg_group_speed', 'density', 'L/W']
    #Фича плотность
    data['density'] = (
        (data['Cut_length'] * data['Cut_quantity'])/
        (data['Width'] * data['Length'])
    )

    #Фича L\W
    data['L/W'] = data['Length'] / data['Width']

    #Кодируем 
    mapping = {'ЛР': 1,
          'ТР': 2}
    
    data['operation_num'] = data['Operation'].replace(mapping)
    data = data[~(data["Cut_length"] > 15000) & (data['Time'] < 500)]

    return data
    
    
    
    
    
    
    
    
