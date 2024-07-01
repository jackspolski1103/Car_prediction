# import numpy as np 
# import pandas as pd
# from category_encoders import TargetEncoder 
# import json 

# def feature_engineering_train(df, dict_name):
#     # por cada elemento de cada columna, se le calcula el promedio y se lo guarda en un diccionario 
#     mean_encoders = {} 
#     mean = df['Precio'].mean()
#     for col in ['Marca', 'Modelo', 'Versión', 'Color', 'Transmisión', 'Tipo de vendedor', 'Tipo de combustible', 'Con cámara de retroceso']:
#         mean_encoders[col] = mean
#         for i in df[col].unique():
#             aux_mean = df[df[col] == i]['Precio'].mean()
#             mean_encoders[i] = aux_mean

#     # guardo el diccionario en un archivo csv donde la primer columna son los keys y la segunda los values 
#     with open(dict_name, 'w') as file:
#         json.dump(mean_encoders, file) 

#     encoder = TargetEncoder(cols = ['Marca', 'Modelo', 'Versión', 'Color', 'Transmisión', 'Tipo de vendedor', 'Tipo de combustible', 'Con cámara de retroceso'], smoothing = 10)
#     final_df = encoder.fit_transform(df, df['Precio'])
#     return final_df


# def feature_engineering_test(df, dict_name):
#     # Levantar el diccionario con los promedios desde el archivo JSON
#     with open(dict_name, 'r') as file:
#         mean_encoders = json.load(file)

#     # Aplicar la transformación a cada columna especificada
#     for col in ['Marca', 'Modelo', 'Versión', 'Color', 'Transmisión', 'Tipo de vendedor', 'Tipo de combustible', 'Con cámara de retroceso']:
#         # Iterar sobre las filas de la columna y aplicar la codificación
#         df[col] = df[col].apply(lambda x: mean_encoders[x] if x in mean_encoders else mean_encoders[col])

#     return df

# def feature_engineering(df, train):
#     # saco las columnas que quiero desestimar 
#     df.drop(columns = ['Título', 'Tipo de carrocería', 'Puertas', 'Moneda', 'Motor'], inplace = True)

#     final_df = df.copy() 

#     # vuelvo a binario las columnas que quedan 
#     final_df['Tracción'] = final_df['Tracción'].apply(lambda x: 1 if x == '4X4' else 0)
#     final_df['Turbo'] = final_df['Turbo'].apply(lambda x: 1 if x == 'SI' else 0)
#     final_df['7plazas'] = final_df['7plazas'].apply(lambda x: 1 if x == 'SI' else 0)

#     dict_name = "./src/ingeniera_datos/mean_encoders.json" 

#     if train:
#         final_df = feature_engineering_train(final_df, dict_name)
#     else:
#         final_df = feature_engineering_test(final_df, dict_name) 

#     precio = final_df.pop('Precio')
#     final_df['Precio'] = precio

#     return final_df.values 

    

import numpy as np 
import pandas as pd
from category_encoders import TargetEncoder 
import json 

def feature_engineering_train(df, dict_name):
    # Crear el encoder y ajustar los datos
    encoder = TargetEncoder(cols=['Marca', 'Modelo', 'Versión', 'Color', 'Transmisión', 'Tipo de vendedor', 'Tipo de combustible', 'Con cámara de retroceso'], smoothing=10)
    final_df = encoder.fit_transform(df, df['Precio'])

    # Guardar el diccionario de encoders en un archivo JSON
    mean_encoders = {col: encoder.mapping[col].to_dict() for col in encoder.cols}
    with open(dict_name, 'w') as file:
        json.dump(mean_encoders, file)

    return final_df


def feature_engineering_test(df, dict_name):
    # Cargar el diccionario con los promedios desde el archivo JSON
    with open(dict_name, 'r') as file:
        mean_encoders = json.load(file)

    # Aplicar la transformación a cada columna especificada
    for col in mean_encoders.keys():
        df[col] = df[col].map(mean_encoders[col]).fillna(df['Precio'].mean())

    return df

def feature_engineering(df, train):
    # Eliminar las columnas que no se quieren usar
    df.drop(columns=['Título', 'Tipo de carrocería', 'Puertas', 'Moneda', 'Motor'], inplace=True)

    final_df = df.copy()

    # Convertir a binario las columnas que quedan
    final_df['Tracción'] = final_df['Tracción'].apply(lambda x: 1 if x == '4X4' else 0)
    final_df['Turbo'] = final_df['Turbo'].apply(lambda x: 1 if x == 'SI' else 0)
    final_df['7plazas'] = final_df['7plazas'].apply(lambda x: 1 if x == 'SI' else 0)

    dict_name = "./src/ingeniera_datos/mean_encoders.json"

    if train:
        final_df = feature_engineering_train(final_df, dict_name)
    else:
        final_df = feature_engineering_test(final_df, dict_name)

    precio = final_df.pop('Precio')
    final_df['Precio'] = precio

    return final_df.values
