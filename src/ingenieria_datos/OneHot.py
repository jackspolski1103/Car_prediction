import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def feature_engineering(df):
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

    df_precio = final_df['Precio']
    final_df.drop(columns = ['Precio'], inplace = True)

    # paso a numpy array
    X = final_df.values
    Y = df_precio.values 
    print(X.shape)
    print(Y.shape)
    return X, Y 

