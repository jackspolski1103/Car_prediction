import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import joblib
from src.train_and_test.test import test

class Model:
    def __init__(self, name, n_neighbors, weights, p , cross_val= False, cv=5):
        self.name = name
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.cross_val = cross_val
        self.cv = cv
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=-1)

    def split_data(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    
    def normalize(self, data):
        # Normalizar con min max y guardar los valores de min y max para cada columna (solo las features que no son binarias)
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        # ver que columnas son binarias
        self.max[self.max == self.min] = self.min[self.max == self.min] + 1
        return (data - self.min) / (self.max - self.min)
    
  


    def train(self,data):
        X, y = self.split_data(data)
        X = self.normalize(X)
        if self.cross_val:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15,25,35],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(self.model, param_grid, cv=kf, n_jobs=-1, verbose=2)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            self.n_neighbors = grid_search.best_params_['n_neighbors']
            self.weights = grid_search.best_params_['weights']
            self.p = grid_search.best_params_['p']

        else:
            self.model.fit(X, y)

    def predict(self, X):
        X = self.normalize(X)
        y = self.model.predict(X)
        return y
    

    def test(self, data):
        X, y = self.split_data(data)
        y_pred = self.predict(X)
        return test(y, y_pred)
    

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
