import pandas as pd
import numpy as np

ionosphere_data = pd.read_csv('ionosphere\ionosphere.data')
adult_data = pd.read_csv('adult\adult.data')
heart_disease_data = pd.read_csv('heart_disease\cleveland.data')
iris_data = pd.read_csv('iris\iris.data')


#Convert the dataset to a NumPy array
ionosphere_np = ionosphere_data.values
adult_data_np = adult_data.values
heart_disease_np = heart_disease_data.values
iris_np = iris_data.values

