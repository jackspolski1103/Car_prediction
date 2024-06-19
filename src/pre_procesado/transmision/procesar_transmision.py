import numpy as np
import pandas as pd 
import Levenshtein as lev 


def preprocesar_transmision(df): 
    tipos = [
    "AUTOMÁTICA",
    "MANUAL",
    "OTRO" 
    ]
    threshold = 5 # cantidad de letras distintas que puede tener como máximo 
    trans = df['Transmisión'].str.split().str[0].str.upper()
    n = len(trans)  
    for i in range(n):
        min_dist = 100
        aux = "" 
        for c in tipos: 
            if pd.isnull(trans[i]) or trans[i] is pd.NA:  
                trans[i] = "OTRO" 
            dist = lev.distance(trans[i], c) 
            if dist < min_dist:
                aux = c  
                min_dist = dist 
        trans[i] = aux 
        if min_dist > threshold:
            trans[i] = "OTRO"
    df['Transmisión'] = trans
    return 

