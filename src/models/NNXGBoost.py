# En esta clase se va a entrenar un modelo de xgboost con input de una capa oculta de la red neuronal
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.train_and_test.test import test
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import joblib
import logging


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
class Model:
    def __init__(self, name, device, input_dim, lr, epocas, batch_size, cv=False, cv_folds=5):
        self.name = name
        self.device = device
        self.input_dim = input_dim
        self.lr = lr
        self.epocas = epocas
        self.batch_size = batch_size
        self.model = SimpleNN(input_dim).to(device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.cv = cv
        self.cv_folds = cv_folds
        self.model_xgboost = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        

    def split_data(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        return X, y

    def normalize(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.max[self.max == self.min] = self.min[self.max == self.min] + 1
        return (data - self.min) / (self.max - self.min)

    def train_NN(self, Metadata_train):
        x_train, y_train = self.split_data(Metadata_train)
        x_train = self.normalize(x_train)
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
    
        for epoch in range(self.epocas):
            self.model.train()
            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]
                y_pred = self.model(x_batch)
                y_pred = y_pred.view(-1)
                loss = self.loss(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            print(f'Epoch {epoch+1}/{self.epocas}, Loss: {loss.item()}')
    
    def get_hidden_layer(self, data):
        X = data
        X = self.normalize(X)
        X= X.astype(np.float32)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        hidden_layer = self.model.relu1(self.model.fc1(X))
        return hidden_layer.detach().cpu().numpy()
    
    def train_xgboost(self, hidden_layer, y_train):
        if self.cv:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 8, 15],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8]
            }
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            grid_search = GridSearchCV(self.model_xgboost, param_grid, cv=kf, n_jobs=-1, verbose=2)
            grid_search.fit(hidden_layer, y_train)
            self.model_xgboost = grid_search.best_estimator_
            self.lr = grid_search.best_params_['learning_rate']
            self.max_depth = grid_search.best_params_['max_depth']
            self.n_estimators = grid_search.best_params_['n_estimators']
            self.subsample = grid_search.best_params_['subsample']
            self.colsample_bytree = grid_search.best_params_['colsample_bytree']
        else:
            self.model_xgboost.fit(hidden_layer, y_train)

    def train(self, data):
        X, y = self.split_data(data)
        self.train_NN(data)
        hidden_layer = self.get_hidden_layer(X)
        self.train_xgboost(hidden_layer, y)


    def predict(self, X):
        hidden_layer = self.get_hidden_layer(X)
        y = self.model_xgboost.predict(hidden_layer)
        return y
    
    def test(self, data):
        X, y = self.split_data(data)
        y_pred = self.predict(X)
        return test(y, y_pred)
    
    def save_model(self, path):
        #guardar la instancia de esta clase
        joblib.dump(self.model_xgboost, path)
        #guardar el modelo de la red neuronal 
        #cambiar el nombre al path a que lo ultimo sea model_nn.pkl
        path = path.split('.')[0] + '_NN.pkl'
        torch.save(self.model, path )


    def load_model(self, path):
        self.model_xgboost = joblib.load(path)
        self.model = torch.load(path + '_NN')
        self.model.eval()
        self.model.to(self.device)

        


    


