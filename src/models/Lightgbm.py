import numpy as np
import pandas as pd
#importar las librerias necesarias para lightgbm 
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import joblib
from src.train_and_test.test import test


class Model:
    def __init__(self, name, n_estimators=100, max_depth=-1, num_leaves=31, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, cross_val=False, cv=5):
        self.name = name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.cross_val = cross_val
        self.cv = cv
        self.model = lgb.LGBMRegressor()

    def split_data(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        return X, y

    def train(self, data):
        X, y = self.split_data(data)
        if self.cross_val:
            param_grid = {
                'num_leaves': [10,31, 50],
                'max_depth': [4, 8,15],
                'learning_rate': [0.1, 0.01, 0.001],
                'n_estimators': [100, 200]
            }
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            grid_search = GridSearchCV(self.model, param_grid, cv=kf, n_jobs=-1, verbose=2)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            self.num_leaves = grid_search.best_params_['num_leaves']
            self.max_depth = grid_search.best_params_['max_depth']
            self.learning_rate = grid_search.best_params_['learning_rate']
            self.n_estimators = grid_search.best_params_['n_estimators']

        else:
            self.model.fit(X, y)

    def predict(self, X):
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


