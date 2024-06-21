import pandas as pd
import numpy as np

def preprocesar_precio(df, dolar):
    # fecha de bajada del dataset: 9/05/2024 
    # cotizacion del dolar: 1030

    monedas = df["Moneda"].copy()
    precios = df["Precio"].values
    for p in range(len(precios)):
        if pd.isnull(precios[p]) or precios[p] is pd.NA:  
            precios[p] = 0
        if monedas[p] == "$":
            precios[p] = precios[p] / dolar  
            # le dejo solo 4 decimales
            precios[p] = round(precios[p], 4)
            monedas[p] = "U$S" 
    df["Precio"] = precios
    df["Moneda"] = monedas
    return df 

def limpiar_datos(df):
    ''' 
    Esta función elimina las filas con valores nulos en la columna precio, las filas que el precio sea 0 y las filas que el precio sea menor a 1000
    tambien elimina las filas con precios mayores a 1000000
    '''
    df = df.dropna(subset=['Precio'])
    df = df[df['Precio'] != 0]
    df = df[df['Precio'] > 500]
    df = df[df['Precio'] < 500000]
    return df



def main():
    df = pd.read_csv('../data/data.csv')
    df = preprocesar_precio(df, 1030)
    df = limpiar_datos(df)
    df.to_csv('../data/Limpio/datos_limpios.csv', index=False)
    return 1

if __name__ == '__main__':
    main()