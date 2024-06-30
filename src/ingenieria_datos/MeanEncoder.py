import numpy as np 
import pandas as pd
from category_encoders import TargetEncoder 


def feature_engineering(df,train = True):
    df_completo = pd.read_csv('./data/Limpio/PreProcesado/completo.csv')
    mean_marcas = df_completo.groupby('Marca')['Precio'].mean()
    mean_modelos = df_completo.groupby('Modelo')['Precio'].mean()
    mean_versiones = df_completo.groupby('Versión')['Precio'].mean()

    # saco las columnas que quiero desestimar 
    df.drop(columns = ['Título', 'Tipo de carrocería', 'Puertas', 'Moneda', 'Motor'], inplace = True)

    encoder = TargetEncoder(cols = ['Marca', 'Modelo', 'Versión', 'Color', 'Tracción', 'Transmisión', 'Tipo de vendedor', 'Tipo de combustible', 'Con cámara de retroceso'], smoothing = 10)
    final_df = encoder.fit_transform(df, df['Precio'])

    # vuelvo a binario las columnas que quedan 
    final_df['Tracción'] = final_df['Tracción'].apply(lambda x: 1 if x == '4X4' else 0)
    final_df['Turbo'] = final_df['Turbo'].apply(lambda x: 1 if x == 'SI' else 0)
    final_df['7plazas'] = final_df['7plazas'].apply(lambda x: 1 if x == 'SI' else 0)
    

    precio = final_df.pop('Precio')
    final_df['Precio'] = precio
    return final_df.values

    
def feature_engineering_test(df):
    