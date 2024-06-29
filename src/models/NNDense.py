import importlib
import torch
import torch.nn as nn 
import numpy as np
import logging
from torch import nn, optim
from src.train_and_test.test import test


class Model:
    def __init__(self, name, device, input_dim, lr,epocas, batch_size):
        self.name = name
        self.model = None
        self.device = device
        self.input_dim = input_dim
        self.lr = lr
        self.epocas = epocas
        self.batch_size = batch_size
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(device)
        
    def split_data(self, data):
        #la ultima columna es la variable a predecir
        #los datos son un array de numpy
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
        
    def train(self,Metadata_train):
        x_train, y_train = self.split_data(Metadata_train)
        x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epocas):
            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]
                y_pred = self.model(x_batch)
                loss = self.loss(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logging.info(f'Epoch {epoch+1}/{self.epocas}, Loss: {loss.item()}')

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(x).cpu().numpy()
    
    def test(self, metadata_test):
        x_test, y_test = self.split_data(metadata_test)
        y_pred = self.predict(x_test)
        return test(y_test, y_pred)
        
            
    def save_model(self,path_to_save):
        #guardar el modelo en path_to_save
        torch.save(self.model, path_to_save)

    def load_model(self,path_to_load):
        #cargar el modelo de path_to_load
        self.model = torch.load(path_to_load)
        
    def load_model(self,path_to_load):
        #cargar el modelo de path_to_load
        self.model = torch.load(path_to_load)

    