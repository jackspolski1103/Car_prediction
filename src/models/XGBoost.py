import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import joblib
from src.train_and_test.test import test

#el modelo de xgboost es un modelo de ensamble de arboles de decision que tiene de entrada 370 columnas y una salida regresora

class Model:
    def __init__(self, name, lr, max_depth, n_estimators,subsample, colsample_bytree , cross_val= False, cv=5):
        self.name = name
        self.lr = lr
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.cross_val = cross_val
        self.cv = cv
        self.model = xgb.XGBRegressor(learning_rate=lr, max_depth=max_depth, n_estimators=n_estimators, subsample=subsample, colsample_bytree=colsample_bytree, n_jobs=-1, random_state=42)

    def split_data(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        return X, y


    def train(self,data):
        X, y = self.split_data(data)
        if self.cross_val:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2, 0.5],
                'max_depth': [3, 8, 10, 15, 20, 25],  
                'subsample': [0.7, 0.8, 0.9, 1.0, 1.2],
                'colsample_bytree': [0.5, 0.8, 0.9, 1.0, 1.2]
            }
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(self.model, param_grid, cv=kf, n_jobs=-1, verbose=2)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            self.lr = grid_search.best_params_['learning_rate']
            self.max_depth = grid_search.best_params_['max_depth']
            self.n_estimators = grid_search.best_params_['n_estimators']
            self.subsample = grid_search.best_params_['subsample']
            self.colsample_bytree = grid_search.best_params_['colsample_bytree']

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


