import numpy as np
import pandas as pd 
import Levenshtein as lev 

def preprocesar_modelos(df):
    modelos = [
        '2008', 
        '208', 
        '3008', 
        '4008', 
        '4Runner', 
        '5008', 
        '500X', 
        'Actyon', 
        'Agile', 
        'Amigo', 
        'Blazer', 
        'Bronco', 
        'Bronco Sport', 
        'C-HR', 
        'C3', 
        'C3 Aircross', 
        'C4', 
        'C4 Aircross', 
        'C4 Cactus', 
        'C5 Aircross', 
        'Captiva', 
        'Captur', 
        'Cayenne', 
        'Cherokee', 
        'Clase E', 
        'Clase GL', 
        'Clase GLA', 
        'Clase GLB', 
        'Clase GLC', 
        'Clase GLE', 
        'Clase GLK', 
        'Clase ML', 
        'Commander', 
        'Compass', 
        'Cooper Countryman', 
        'Corolla Cross',  
        'Coupe', 
        'CR-V', 
        'Creta', 
        'Crossfox', 
        'Defender', 
        'Discovery', 
        'DS3', 
        'DS7', 
        'DS7 Crossback', 
        'Duster', 
        'Duster Oroch', 
        'E-tron', 
        'Ecosport', 
        'Emgrand X7 Sport', 
        'Equinox', 
        'Evoque', 
        'Explorer', 
        'F-PACE', 
        'Feroza', 
        'Forester', 
        'Freelander', 
        'Galloper', 
        'Grand Blazer', 
        'Grand Cherokee', 
        'Grand Santa Fé', 
        'Grand Vitara', 
        'H1', 
        'H6', 
        'Hilux', 
        'HR-V', 
        'Jimny', 
        'Jolion', 
        'Journey', 
        'Kangoo', 
        'Kicks', 
        'Koleos', 
        'Kona', 
        'Kuga', 
        'Land Cruiser', 
        'LX', 
        'Macan', 
        'ML', 
        'Mohave', 
        'Montero', 
        'Murano', 
        'Musso', 
        'Mustang', 
        'Myway', 
        'Nativa', 
        'Nivus', 
        'NX', 
        'Outback', 
        'Outlander', 
        'Panamera', 
        'Pathfinder', 
        'Patriot', 
        'Pilot', 
        'Pulse', 
        'Q2', 
        'Q3', 
        'Q3 Sportback', 
        'Q5', 
        'q5 sportback', 
        'Q7', 
        'Q8', 
        'Range Rover', 
        'RAV4', 
        'Renegade', 
        'Rodeo', 
        'S2', 
        'S5', 
        'Samurai', 
        'Sandero', 
        'Santa Fe', 
        'Seltos', 
        'Serie 4', 
        'Sorento', 
        'Soul', 
        'Spin', 
        'Sportage',
        'Stelvio', 
        'Suran', 
        'SW4', 
        'T-Cross', 
        'Taos', 
        'Terios', 
        'Terrano II', 
        'Territory', 
        'Tiggo', 
        'Tiggo 2', 
        'Tiggo 3', 
        'Tiggo 4', 
        'Tiggo 4 Pro', 
        'Tiggo 5', 
        'Tiggo 8 Pro', 
        'Tiguan', 
        'Tiguan Allspace', 
        'Touareg', 
        'Tracker', 
        'Trailblazer', 
        'Trooper', 
        'Tucson', 
        'UX', 
        'Veracruz', 
        'Vitara', 
        'Wrangler', 
        'X-Terra', 
        'X-Trail', 
        'X1', 
        'X2', 
        'X25', 
        'X3', 
        'X35', 
        'X4', 
        'X5', 
        'X55', 
        'X6', 
        'X70', 
        'XC40', 
        'XC60'
    ]
    threshold = 8 # cantidad de letras distintas que puede tener como máximo 
    modelos = [m.upper() for m in modelos] 
    modelo = df['Modelo'].str.upper()
    n = len(modelo)  
    for i in range(n):
        min_dist = 100
        aux = "" 
        for c in modelos:
            # print("color[i]: ", color[i]) 
            # print("c: ", c)  
            if pd.isnull(modelo[i]) or  modelo[i] is pd.NA:  
                modelo[i] = "OTRO" 
            dist = lev.distance(modelo[i], c) 
            if dist < min_dist:
                aux = c  
                min_dist = dist 
        modelo[i] = aux 
        if min_dist > threshold:
            modelo[i] = "OTRO"
    df['Modelo'] = modelo
    return 


# prueba 
df = pd.read_csv("../../data/data.csv")
preprocesar_modelos(df)
modelos = df['Modelo'].value_counts() 
# ordenarlos alfabeticamente
modelos = modelos.sort_index()
modelos.to_csv("modelos_procesados.csv")
