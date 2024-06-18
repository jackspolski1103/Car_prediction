import pandas as pd
import numpy as np
import re


def encontrar_numero_decimal(texto):

    #buscar formato con un decimal y un punto
    patron = r'\d+\.\d'
    matches = re.findall(patron, texto)
    if len(matches) > 0:
        #devolver el menor
        result = min([float(i) for i in matches])
        if result<10:
            return result
         
    else:
        #buscar numero entero
        patron = r'\d+'
        matches = re.findall(patron, texto)
        if len(matches) > 0:
            #devolver el menor
            result = min([float(i) for i in matches])
            if result<10:
                return result

        else:
            return 0.0
    return 0.0

def preprocesar_motor(df):
    motores= df['Motor'].str.upper()
    #cambiar las , por .
    motores = motores.str.replace(',','.')
    # Si hay una posicion vacia, rellenar con 0
    motores = motores.fillna('0')

    #hacemos la columna cilindrada
    df['Cilindrada'] = motores.apply(lambda x: encontrar_numero_decimal(x))
    #hacemos la columna potencia
    #QUEDA PENDIENTE
    return df

        