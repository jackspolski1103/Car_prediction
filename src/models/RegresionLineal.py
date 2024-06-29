import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.train_and_test.test import test
import joblib

class Model:
    def __init__(self, name, device, input_dim):
        self.name = name
        self.model = None

    def normalize(self, data):
        #Normalizar con min max y guardar los valores de min y max para cada columna
        #data es un numpy array
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        # ver el caso en que el maximo y el minimo sean iguales
        self.max[self.max == self.min] = self.min[self.max == self.min] + 1
        return (data - self.min) / (self.max - self.min)
    

    def split_data(self, data):
        #la ultima columna es la variable a predecir
        #los datos son un array de numpy
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
        

    def train(self,Metadata_train):
        # Metadata_train = self.normalize(Metadata_train)
        x_train, y_train = self.split_data(Metadata_train)
        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        return self.model

    def predict(self, x):
        #normalizar x usando self.min y self.max (x no incluye la variable a predecir)
        # x = (x - self.min[:-1]) / (self.max[:-1] - self.min[:-1])
        prediction = self.model.predict(x)
        #desnormalizar la prediccion
        # return prediction * (self.max[-1] - self.min[-1]) + self.min[-1]
        return prediction
    


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

    




        
    
    