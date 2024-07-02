import numpy as np
from sklearn.decomposition import PCA
from src.ingenieria_datos.OneHot import feature_engineering as one_hot
import joblib

#hacer pca con los datos de entrada de train y guardar las componentes principales para poder hacer la transformacion en test
def train_pca(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca

def feature_engineering(data, train=True):
    if train:
        data = np.load('./results/OneHot/metadata_train.npy', allow_pickle=True)
    else:
        data = np.load('./results/OneHot/metadata_test.npy', allow_pickle=True)

    X = data[:, :-1]
    y = data[:, -1]

    if train:
        pca = train_pca(X, 20)
        X = pca.transform(X)
        # Guardar el modelo PCA en lugar de solo las componentes principales
        joblib.dump(pca, './src/ingenieria_datos/pca_model.pkl')
        return np.column_stack((X, y))
    else:
        # Cargar el modelo PCA guardado
        pca = joblib.load('./src/ingenieria_datos/pca_model.pkl')
        X = pca.transform(X)
        return np.column_stack((X, y))
    

