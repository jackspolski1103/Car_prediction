import numpy as np 
import pandas as pd 


def preprocesar_kilometro(df):
    años = df["Año"].copy()
    kilometros = df["Kilómetros"].values
    threshold = 80000 
    promedio = 20000

    for k in range(len(kilometros)):
        km = kilometros[k].split(" ")[0]
        antiguedad = np.abs(2025 - años[k]) 
        if pd.isnull(kilometros[k]) or kilometros[k] is pd.NA:
            if años[k] == 2024:
                kilometros[k] = 0.0 
            kilometros[k] = promedio * antiguedad 
        else: 
            relacion = float(km) / antiguedad
            if relacion > threshold:
                if años[k] == 2024:
                    kilometros[k] = 0.0 
                kilometros[k] = promedio * antiguedad 
            else:
                kilometros[k] = km
    df["Kilómetros"] = kilometros
