import numpy as np
import pandas as pd 
import Levenshtein as lev 


def preprocesar_combustible(df): 
    tipos = [
        "NAFTA",
        "DIÉSEL",
        "GNC",
        "HÍBRIDO",
        "HÍBRIDO/NAFTA",
        "HÍBRIDO/DIESEL",
        "NAFTA/GNC",
        "ELÉCTRICO",
        "OTRO_COMBUSTIBLE" 
    ]
    threshold = 3 # cantidad de letras distintas que puede tener como máximo 
    combustible = df['Tipo de combustible'].str.split().str[0].str.upper()
    n = len(combustible)  
    for i in range(n):
        min_dist = 100
        aux = "" 
        for c in tipos: 
            if pd.isnull(combustible[i]) or combustible[i] is pd.NA:  
                combustible[i] = "OTRO_COMBUSTIBLE" 
            dist = lev.distance(combustible[i], c) 
            if dist < min_dist:
                aux = c  
                min_dist = dist 
        combustible[i] = aux 
        if min_dist > threshold:
            combustible[i] = "OTRO_COMBUSTIBLE"
    df['Tipo de combustible'] = combustible
    return 

