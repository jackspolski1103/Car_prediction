import numpy as np
import pandas as pd 
import Levenshtein as lev 


def preprocesar_vendedor(df): 
    tipos = [
        "TIENDA",
        "CONCESIONARIA",
        "PARTICULAR",
        "OTRO"
    ]
    threshold = 3 # cantidad de letras distintas que puede tener como máximo 
    vendedor = df['Tipo de vendedor'].str.split().str[0].str.upper()
    n = len(vendedor)  
    for i in range(n):
        min_dist = 100
        aux = "" 
        for c in tipos: 
            if pd.isnull(vendedor[i]) or vendedor[i] is pd.NA:  
                vendedor[i] = "OTRO" 
            dist = lev.distance(vendedor[i], c) 
            if dist < min_dist:
                aux = c  
                min_dist = dist 
        vendedor[i] = aux 
        if min_dist > threshold:
            vendedor[i] = "OTRO"
    df['Tipo de vendedor'] = vendedor
    return 

