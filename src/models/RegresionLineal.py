import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.train_and_test.test import test
import joblib

class Model:
    def __init__(self, name, device, input_dim):
        self.name = name
        self.model = None

    def split_data(self, data):
        #la ultima columna es la variable a predecir
        #los datos son un array de numpy
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
        

    def train(self,Metadata_train):
        x_train, y_train = self.split_data(Metadata_train)
        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        return self.model

    def predict(self, x):
        return self.model.predict(x)
    
    def test(self, metadata_test):
        x_test, y_test = self.split_data(metadata_test)
        y_pred = self.predict(x_test)
        return test(y_test, y_pred)
            
    def save_model(self,path_to_save):
        #guardar el modelo en path_to_save
        joblib.dump(self.model, path_to_save)

    def load_model(self,path_to_load):
        #cargar el modelo de path_to_load
        self.model = joblib.load(path_to_load)

    




        
    
    