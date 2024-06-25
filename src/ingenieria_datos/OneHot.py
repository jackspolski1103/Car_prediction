import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def feature_engineering(df):
    # # drop columna Título, Tipo de carrocería, Puertas, Moneda, Motor

    # df.drop(columns=['Título', 'Tipo de carrocería', 'Puertas', 'Moneda', 'Motor'], inplace=True)
    # # OneHotEncoding
    # # OneHotEncoding para las columnas Marca, Modelo, Versión, Color, Tracción, Transmisión, Tipo de vendedor, Tipo de combustible, Con cámaras de retroceso, Turbo, 7plazas

    # Column_transformer = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), ['Marca', 'Modelo', 'Versión', 'Color', 'Tracción', 'Transmisión', 'Tipo de vendedor', 'Tipo de combustible', 'Con cámara de retroceso', 'Turbo', '7plazas'])], remainder='passthrough')
    # tranformed_data = Column_transformer.fit_transform(df)
    # df = pd.DataFrame(tranformed_data, columns=Column_transformer.get_feature_names_out())
    return df


    # OneHotEncoding para la columna Modelo

