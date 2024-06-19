import numpy as np
import pandas as pd 
import Levenshtein as lev 



def preprocesar_marcas(df): 
    marcas = [
         'PEUGEOT',
         'TOYOTA', 
         'FIAT', 
         'DS', 
         'SSANGYONG', 
         'CHEVROLET',
         'CITROËN',
         'ISUZU',
         'FORD',
         'RENAULT',
         'PORSCHE',
         'JEEP',
         'MERCEDES-BENZ', 
         'MINI',
         'HONDA',
         'HYUNDAI',
         'VOLKSWAGEN',
         'LAND',
         'AUDI',
         'GEELY',
         'JAGUAR',
         'DAIHATSU',
         'SUBARU',
         'SUZUKI',
         'HAVAL',
         'DODGE',
         'NISSAN',
         'LEXUS',
         'KIA',
         'MITSUBISHI',
         'LIFAN',
         'JAC',
         'BMW',
         'ALFA',
         'CHERY',
         'BAIC',
         'JETOUR',
         'VOLVO'
         ]
    threshold = 2 # cantidad de letras distintas que puede tener como máximo 
    marca = df['Marca'].str.split().str[0].str.upper()
    n = len(marca)  
    for i in range(n):
        min_dist = 100
        aux = "" 
        for c in marcas:
            # print("color[i]: ", color[i]) 
            # print("c: ", c)  
            if pd.isnull(marca[i]) or  marca[i] is pd.NA:  
                marca[i] = "OTRO" 
            dist = lev.distance(marca[i], c) 
            if dist < min_dist:
                aux = c  
                min_dist = dist 
        marca[i] = aux 
        if min_dist > threshold:
            marca[i] = "OTRO"
    df['Marca'] = marca
    return 

