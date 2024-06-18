import numpy as np
import pandas as pd 
import Levenshtein as lev 
import re 


def calculate_min_dist(value, list):  
    min_dist = 100 
    index = 0 
    for i in range(len(list)): 
        dist = lev.distance(value, list[i]) 
        if dist < min_dist:
            min_dist = dist 
            index = i
    return index


def preprocesar_motor(df): 
    motores = [
        "1",
        "1.2",
        "1.3", 
        "1.4",
        "1.5",
        "1.6",
        "1.8",
        "1.9",
        "2",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "2.8",
        "2.9",
        "3",
        "3.1",
        "3.2",
        "3.3",
        "3.5",
        "3.6",
        "3.7",
        "3.8",
        "4",
        "4.2",
        "4.3",
        "4.4",
        "4.5",
        "4.6",
        "4.7",
        "4.8",
        "5",
        "5.2",
        "5.5",
        "5.7",
        "6",
        "6.1",
        "6.4",
        "8",
        "V6",
        "V8",
    ]
    threshold = 4 # cantidad de letras distintas que puede tener como máximo 
    motor = df['Motor'].str.upper()
    n = len(motor)  
    for i in range(n):
        # recorro cada motor 
        min_dist = 100
        aux_label = "" 

        for m in motores:
            # recorro cada motor standarizado 
            if pd.isnull(motor[i]) or motor[i] is pd.NA:  
                motor[i] = "OTRO" 
            
            # si el motor no esta vacio: 
            motor[i] = motor[i].replace(",", ".") 
            motor[i] = re.sub(r'\.0(?!\d)', '', motor[i]) 
            descripcion = motor[i].split() 
            idx = calculate_min_dist(m, descripcion) 
            dist = lev.distance(descripcion[idx], m)   
            if dist < min_dist:
                aux_label = m 
                min_dist = dist 
        motor[i] = aux_label  
        if min_dist >= threshold:
            motor[i] = "OTRO"
    df['Motor'] = motor
    return 

prueba = [ 
    "250TSI 1.4 150 CV",
    "3.0 TURBO INTERCOOLER",
    "200TSI 1.0 TIP",
    "1,6 Turbo", 
    "1,0TSI", 
    "3",
    "2.0T 198 MILD HYBRID" 
]
df = pd.DataFrame(prueba, columns=['Motor']) 
preprocesar_motor(df)
print(df['Motor'].value_counts()) 