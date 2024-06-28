import numpy as np 
import pandas as pd 

def preprocesar_camaraRetroceso(df):
    camara_ret = df["Con cámara de retroceso"].str.upper()
    for i in range(len(camara_ret)):
        if pd.isnull(camara_ret[i]) or camara_ret[i] is pd.NA:
            camara_ret[i] = "NO INDICA"
    df["Con cámara de retroceso"] = camara_ret
    return 
