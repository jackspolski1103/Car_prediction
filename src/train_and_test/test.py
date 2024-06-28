import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error, mean_squared_log_error, median_absolute_error


def test(y_test, y_pred):
    #armar un pandas dataframe con las metricas (para regresion)
    #calcular las metricas
    #devolver el dataframe

    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'explained_variance_score': explained_variance_score(y_test, y_pred),
        'max_error': max_error(y_test, y_pred),
        'mean_squared_log_error': mean_squared_log_error(y_test, y_pred),
        'median_absolute_error': median_absolute_error(y_test, y_pred)
    }
    #pasa el diccionario a un dataframe
    metrics = pd.DataFrame(metrics, index=[0])
    return metrics


    

    
