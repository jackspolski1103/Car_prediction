import logging
import importlib
import os, random
from os.path import join, split, exists
import pandas as pd
import numpy as np


from src.utils.files import create_result_folder



logger = logging.getLogger(__name__)


def run_experiment(model_params, data_params):
    """
    Este ejemplo es para un caso en el cual solamente entrenemos y evaluemos un modelo
    basado en redes. La idea sera modificar estos pasos para que se ajusten a las necesidades
    de lo que cada uno vaya a proponer como modelo. Podria mantenerse esto mismo y simplemente
    complejizar las subfunciones, o bien modificar esta funcion para que se ajuste a un flujo de
    trabajo distinto, eliminando o agregando pasos.
    """
    # Bajar dataset train
    data_train = pd.read_csv('./data/Limpio/PreProcesado/train.csv')
    # Bajar dataset test
    data_test = pd.read_csv('./data/Limpio/PreProcesado/test.csv')


    context = {}
    # create result folder
    

    context['save_path'] = create_result_folder(data_params.name, model_params.name)
        
    # Si en la carpeta \data_paramas.name ya existe el archivo train.csv, no llama a la funcion get_ingenieria_datos y baja el archivo train.csv
    #el archivo metadata deberia estar en results/data_params.name
    path = join('results', data_params.name, 'metadata_train.npy')
    if not exists(path):
        metadata_train = get_ingenieria_datos(data_params, data_train, train = True)
        metadata_test = get_ingenieria_datos(data_params, data_test, train = False)
        #descargar metadata_train que es un numpy array
        np.save(join('results', data_params.name, 'metadata_train.npy'), metadata_train)
        np.save(join('results', data_params.name, 'metadata_test.npy'), metadata_test)



        # metadata_train.to_csv(path, index=False)
        # metadata_test.to_csv(join('results', data_params.name, 'metadata_test.csv'), index=False)
    else:
        print("Path: ", path)
        metadata_train = np.load(path, allow_pickle = True)
        metadata_test = np.load(join('results', data_params.name, 'metadata_test.npy'), allow_pickle = True)
    
    
    print('metadata_train', metadata_train.shape)
    print('metadata_test', metadata_test.shape)
    # load model
    model = load_model(model_params)
    
    # train
    model.train(metadata_train)

    # test
    results = model.test(metadata_test)
    
    # save model
    model_path = join(context['save_path'], 'model.pkl')
    model.save_model(model_path)
    print('model saved')
    # test
    # save results
    results.to_csv(join(context['save_path'], 'results.csv'), index=False)


    logger.info('Experiment completed')
    return 

def load_model(params):
    model_module = importlib.import_module(f'src.models.{params.name}')
    #crear una instancia de la clase que se llama igual que el archivo y recibe como parametro params
    # tranformar params en una lista
    model = model_module.Model(*params.params)
    return model

   

def get_ingenieria_datos(data_params, data, train=True):
    #esta funcion entra a la carpeta de ingenieria de datos y ejecuta el script que tenga el nombre de data_params.name la funcion que se ejecuta es la que se llama feauture_engineering
    #devuelve el dataframe que se genero en esa funcion
    data_module = importlib.import_module(f'src.ingenieria_datos.{data_params.name}')
    metadata_train = data_module.feature_engineering(data,train)
    return metadata_train
