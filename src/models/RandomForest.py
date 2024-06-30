import numpy as np
import pandas as pd
#importar las librerias necesarias para random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import joblib
from src.train_and_test.test import test




class Model:
    def __init__(self, name, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, cross_val=False, cv=5):
        self.name = name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.cross_val = cross_val
        self.cv = cv
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            n_jobs=-1, 
            random_state=42
        )

    def split_data(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        return X, y

    def train(self, data):
        X, y = self.split_data(data)
        if self.cross_val:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 8, 15, 25, 45],
                'min_samples_split': [2, 5, 10, 25],
                'min_samples_leaf': [1, 2, 4, 10]
            }
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            grid_search = GridSearchCV(self.model, param_grid, cv=kf, n_jobs=-1, verbose=2)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            self.n_estimators = grid_search.best_params_['n_estimators']
            self.max_depth = grid_search.best_params_['max_depth']
            self.min_samples_split = grid_search.best_params_['min_samples_split']
            self.min_samples_leaf = grid_search.best_params_['min_samples_leaf']
        else:
            self.model.fit(X, y)

    def predict(self, X):
        y = self.model.predict(X)
        return y

    def test(self, data):
        X, y = self.split_data(data)
        y_pred = self.predict(X)
        return test(y, y_pred)

    def save_model(self, path_to_save):
        joblib.dump(self.model, path_to_save)

    def load_model(self, path_to_load):
        self.model = joblib.load(path_to_load)

