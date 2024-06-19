import numpy as np
import pandas as pd 


def preprocesar_año(df):
    # Años validos entre 1900 y 2024
    #si es menor a 100  y mayor a 24 se le suma 1900
    df['Año'] = df['Año'].apply(lambda x: x+1900 if x<100 and x>24 else x)
    # si es menor a 1900 y esta entre 0 y 24 se le suma 2000
    df['Año'] = df['Año'].apply(lambda x: x+2000 if x<24 and x>0 else x)
    #si es menor a 1900 se le pone 1900
    df['Año'] = df['Año'].apply(lambda x: 1900 if x<1900 else x)
    #si es mayor a 2024 se le pone 2024
    df['Año'] = df['Año'].apply(lambda x: 2024 if x>2024 else x)

    return df





