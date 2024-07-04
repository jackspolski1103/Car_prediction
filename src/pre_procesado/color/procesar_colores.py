import numpy as np
import pandas as pd 
import Levenshtein as lev 


def preprocesar_colores(df): 
    colores = [
    "BLANCO",
    "NEGRO",
    "AZUL",
    "ROJO",
    "GRIS",
    "PLATEADO", 
    "VERDE",
    "AMARILLO",
    "VIOLETA",
    "BEIGE",
    "CELESTE",
    "DORADO",
    "NARANJA",
    "BORDO",
    "MARRON",
    "OTRO_COLOR" 
    ]
    threshold = 5 # cantidad de letras distintas que puede tener como máximo 
    color = df['Color'].str.split().str[0].str.upper()
    n = len(color)  
    for i in range(n):
        min_dist = 100
        aux = "" 
        for c in colores:
            # print("color[i]: ", color[i]) 
            # print("c: ", c)  
            if pd.isnull(color[i]) or color[i] is pd.NA:  
                color[i] = "OTRO_COLOR" 
            dist = lev.distance(color[i], c) 
            if dist < min_dist:
                aux = c  
                min_dist = dist 
        color[i] = aux 
        if min_dist > threshold:
            color[i] = "OTRO_COLOR"
    df['Color'] = color
    return 

