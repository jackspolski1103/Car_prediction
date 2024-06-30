import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tqdm import tqdm


def one_hot_encoder(df):
    onehot = OneHotEncoder()
    df.drop(columns = ['Título', 'Tipo de carrocería', 'Puertas', 'Moneda', 'Motor'], inplace=True)
    columns_to_keep = df[['Año','Kilómetros', 'Precio', 'Cilindrada', 'Tracción', 'Turbo', '7plazas']]
    arrays = onehot.fit_transform(df[['Marca', 'Modelo', 'Versión', 'Color', 'Tracción', 'Transmisión', 'Tipo de vendedor', 'Tipo de combustible', 'Con cámara de retroceso']]).toarray() 
    new_features = onehot.categories_
    new_features = np.concatenate(new_features)
    encoded_df = pd.DataFrame(arrays, columns=new_features)
    final_df = pd.concat([columns_to_keep.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # a la columna "Traccion" donde dice "4X4" ponerle 1 y a las demas 0 
    final_df['Tracción'] = final_df['Tracción'].apply(lambda x: 1 if x == '4X4' else 0)
    final_df['Turbo'] = final_df['Turbo'].apply(lambda x: 1 if x == 'SI' else 0)
    final_df['7plazas'] = final_df['7plazas'].apply(lambda x: 1 if x == 'SI' else 0)

    # poner la columna precio como la ultima columna del final_df 
    precio = final_df.pop('Precio')
    final_df['Precio'] = precio
    return final_df
    

def complete_rows(df, df_completo): 
    # Inicializar un DataFrame vacío con las mismas columnas que df_completo
    df_final = pd.DataFrame(columns=df_completo.columns)
    
    # Recorrer cada fila del DataFrame df
    for index, row in tqdm(df.iterrows(), desc ='Completando filas'): 
        marca = row['Marca']
        modelo = row['Modelo']
        ano = row['Año']
        version = row['Versión']
        color = row['Color']
        combustible = row['Tipo de combustible']
        transmision = row['Transmisión']
        km = row['Kilómetros']
        vendedor = row['Tipo de vendedor']
        camara = row['Con cámara de retroceso']
        cilindrada = row['Cilindrada']
        traccion = row['Tracción']
        turbo = row['Turbo']
        plazas = row['7plazas']
        precio = row['Precio']

        new_row = {col: 0 for col in df_completo.columns}  # Inicializar una fila nueva con ceros

        # Asignar valores a la nueva fila
        new_row[marca] = 1 
        new_row[modelo] = 1
        new_row['Año'] = ano
        new_row[version] = 1
        new_row[color] = 1
        new_row[combustible] = 1
        new_row[transmision] = 1
        new_row['Kilómetros'] = km
        new_row[vendedor] = 1
        new_row[camara] = 1
        new_row['Cilindrada'] = cilindrada
        new_row['Tracción'] = 1 if traccion == '4X4' else 0
        new_row['Turbo'] = 1 if turbo == 'SI' else 0
        new_row['7plazas'] = 1 if plazas == 'SI' else 0
        new_row['Precio'] = precio

        new_row = pd.DataFrame([new_row], columns=df_completo.columns)
        df_final = pd.concat([df_final, new_row], ignore_index = True)
    return df_final.values 

def feature_engineering(df):
    df_completo = pd.read_csv('./data/Limpio/PreProcesado/completo.csv')
    df_completo = one_hot_encoder(df_completo)

    df_final = complete_rows(df, df_completo)    
        
    return df_final

