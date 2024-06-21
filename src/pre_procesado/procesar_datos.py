import numpy as np 
import pandas as pd 

# import all preprocesing functions
from año.procesar_año                 import preprocesar_año
from camara_ret.procesar_camara_ret   import preprocesar_camaraRetroceso
from color.procesar_colores           import preprocesar_colores
from combustible.procesar_combustible import preprocesar_combustible
from kilometro.procesar_kilometro     import preprocesar_kilometros
from marca.procesar_marcas            import preprocesar_marcas
from modelo.procesar_modelos          import preprocesar_modelos
from motor.procesar_motor             import preprocesar_motor
from plaza.procesar_7plazas           import preprocesar_7plazas
from precio.procesar_precio           import preprocesar_precio   
from traccion.procesar_traccion       import preprocesar_tracciones 
from transmision.procesar_transmision import preprocesar_transmision 
from turbo.procesar_turbo             import preprocesar_turbo
from vendedor.procesar_vendedor       import preprocesar_vendedor
from version.procesar_versiones       import preprocesar_versiones


def preprocesar_datos(df):
    dolar_hoy = 1030 
    preprocesar_marcas(df)
    preprocesar_modelos(df)
    preprocesar_año(df)
    preprocesar_colores(df)
    preprocesar_camaraRetroceso(df)
    preprocesar_combustible(df)
    preprocesar_motor(df)
    preprocesar_tracciones(df)
    preprocesar_transmision(df)
    preprocesar_vendedor(df)
    preprocesar_kilometros(df)
    preprocesar_precio(df, dolar_hoy) 
    preprocesar_turbo(df)
    preprocesar_7plazas(df)
    preprocesar_versiones(df)
    return df 


data_file = '../../data/data.csv'
df = pd.read_csv(data_file) 
preprocesar_datos(df)
print(df.head()) 






