import numpy as np 
import pandas as pd

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

