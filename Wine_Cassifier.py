import pandas as pd 
import sklearn as sk
import matplotlib.pyplot as plt

# Load wine dataset
from sklearn.datasets import load_wine
data = load_wine()

features = data.data
feature_names = data.feature_names
target = data.target

dataset=pd.DataFrame(data = features , columns=feature_names)
dataset['target']= target
print(dataset)

