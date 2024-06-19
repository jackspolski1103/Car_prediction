import re
import Levenshtein as lev 
import numpy as np
import pandas as pd

def leer_archivo_texto(archivo):
    with open(archivo, 'r') as file:
        lines = file.readlines()
        lines = [line.strip().upper() for line in lines]        
    return lines

def crear_diccionario_versiones():
        marcas = [
         'PEUGEOT',
         'TOYOTA', 
         'FIAT', 
         'DS', 
         'SSANGYONG', 
         'CHEVROLET',
         'CITROEN',
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
         'DAIHATSU',
         'SUBARU',
         'SUZUKI',
         'HAVAL',
         'DODGE',
         'NISSAN',
         'LEXUS',
         'KIA',
         'MITSUBISHI',
         'JAC',
         'BMW',
         'ALFA_ROMEO',
         'CHERY',
         'BAIC',
         'VOLVO'
         ]
        # dentro de la carpeta info hay MARCA.txt con las versiones de cada marca
        diccionario_versiones = {}
        for marca in marcas:
            print(marca)
            archivo = f'./version/info/{marca}.txt'
            print(archivo)
            diccionario_versiones[marca] = leer_archivo_texto(archivo)
        return diccionario_versiones



def preprocesar_versiones(df):

    diccionario_versiones = crear_diccionario_versiones()
    threshold = 2
    #cargar versiones y pasar a mayusculas
    versiones = df['Versión'].str.upper().str.strip()
    new_versiones = []
    marcas = df['Marca'].str.upper().str.strip()
    #iterar sobre las versiones, separar por espacios y buscar coincidencias
    for i in range(len(versiones)):
        min_dist = 100
        aux = ""
        for marca in diccionario_versiones:
            if marcas[i] == marca:
                for j in range(len(versiones[i])):
                    for version in diccionario_versiones[marca]:
                        dist = lev.distance(version, versiones[i][j])
                        if dist < min_dist:
                            aux = version
                            min_dist = dist
                if min_dist > threshold:
                    aux = "OTRO"
        if aux == "":
            aux = "OTRO"
        new_versiones.append(aux)

    df['Versión'] = new_versiones
    return df

    




