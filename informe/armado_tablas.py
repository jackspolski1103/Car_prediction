#en la carpeta results/data_params.name/model_params.name se guardan los resultados
#levantar los archivos results/data_params.name/model_params.name/results.csv por data_params.name 
# y armar una tabla donde una columna sea el nombre del modelo y las otras columnas sean las metricas

import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np

def armar_tabla(data_params_name):
    path = '../results/' + data_params_name
    #obtener los nombres de los modelos
    modelos = [f for f in listdir(path) if not isfile(join(path, f))]
    #dentro de la carpeta de cada modelo se encuentran los archivos results.csv
    #levantar los archivos results.csv por modelo
    #armar una tabla donde una columna sea el nombre del modelo y las otras columnas sean las metricas
    df = pd.DataFrame()
    for modelo in modelos:
        #levantar el archivo results.csv
        results = pd.read_csv(join(path, modelo, 'results.csv'))
        #agregar una columna con el nombre del modelo
        results['modelo'] = modelo
        #agregar las metricas a la tabla
        df = pd.concat([df, results], axis = 0)
    return df

def fromDF_to_latex(df):
    #convertir la tabla a formato latex
    return df.to_latex()
    
