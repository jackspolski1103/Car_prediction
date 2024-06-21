import pandas as pd
import numpy as np

def limpiar_datos(df):
    ''' 
    Esta función elimina las filas con valores nulos en la columna precio, las filas que el precio sea 0 y las filas que el precio sea menor a 1000
    tambien elimina las filas con precios mayores a 1000000
    '''
    df = df.dropna(subset=['Precio'])
    df = df[df['Precio'] != 0]
    df = df[df['Precio'] > 500]
    df = df[df['Precio'] < 1000000]
    return df




    #Eliminar las filas con valores nulos en precio