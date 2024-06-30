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


class Model:
    def __init__(self, name, device, input_dim, lr, epocas, batch_size, lr_xgboost, max_depth, n_estimators, subsample, colsample_bytree, cross_val=False, cv=5):
        self.name = name
        self.device = device
        self.input_dim = input_dim
        self.lr = lr
        self.epocas = epocas
        self.batch_size = batch_size
        self.lr_xgboost = lr_xgboost
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.cross_val = cross_val
        self.cv = cv
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(device)
        self.model_xgboost = xgb.XGBRegressor(learning_rate=lr_xgboost, max_depth=max_depth, n_estimators=n_estimators, subsample=subsample, colsample_bytree=colsample_bytree, n_jobs=-1, random_state=42)

    def split_data(self, data):
        # la ultima columna es la variable a predecir
        # los datos son un array de numpy
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    
    def normalize(self, data):
        # Normalizar con min max y guardar los valores de min y max para cada columna
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.max[self.max == self.min] = self.min[self.max == self.min] + 1
        return (data - self.min) / (self.max - self.min)

    def train_nn(self, Metadata_train):
        x_train, y_train = self.split_data(Metadata_train)
        x_train = self.normalize(x_train)
        x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epocas):
            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]
                y_pred = self.model(x_batch)
                loss = self.loss(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logging.info(f'Epoch {epoch + 1}/{self.epocas}, Loss: {loss.item()}')



