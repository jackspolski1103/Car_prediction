import pandas as pd
import numpy as np

def split_datos(df):
    ''' 
    Esta función divide el dataframe en dos dataframes, uno con los datos de entrenamiento y otro con los datos de prueba
    La elección de los datos de entrenamiento y prueba se hace de forma aleatoria
    '''
    train = df.sample(frac=0.9, random_state=0)
    test = df.drop(train.index)
    return train, test


    