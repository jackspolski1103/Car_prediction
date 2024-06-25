import numpy as np
import pandas as pd

class Model:
    def __init__(self, name, device, input_dim):
        self.name = name
        self.device = device
        self.input_dim = input_dim
        self.model = None

    def train(self,Metadata_train):
        # from sklearn.linear_model import LinearRegression
        # self.model = LinearRegression()
        # self.model.fit(X, y)
        return 10
    def save(self,a):
        return 20
    