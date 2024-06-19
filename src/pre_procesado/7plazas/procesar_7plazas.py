import pandas as pd
import numpy as np

def read_txt_file(file):
    with open(file, 'r') as file:
        lines = file.readlines()
        lines = [line.upper() for line in lines]
        #sacar solo los saltos de linea sin sacar los espacios
        lines = [line.replace('\n', '') for line in lines]    
    return lines


def buscar_palabra(palabra, texto):
    return palabra in texto


def preprocesar_7plazas(df):

    plazas=read_txt_file('./7plazas.txt')
    versiones = df['Versión'].str.upper()
    titulos = df['Título'].str.upper()
    #cambiar todos los nan por 'OTRO'
    versiones = versiones.fillna('OTRO')
    titulos = titulos.fillna('OTRO')
    asientos_list = []
    for version, titulo in zip(versiones, titulos):
        asiento_ = 'NO'
        for t in plazas:
            if buscar_palabra(t, version) or buscar_palabra(t, titulo):
                asiento_ = 'SI'
        asientos_list.append(asiento_)
    df['7plazas'] = asientos_list
    return df