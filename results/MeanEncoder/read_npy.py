import numpy as np 
import pandas as pd

# read numpy file and put it into a pandas dataframe
file_path = 'metadata_train.npy' 
data = np.load(file_path)
df = pd.DataFrame(data)
df.to_csv('data.csv', index = False)

