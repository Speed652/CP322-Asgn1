import pandas as pd

ionosphere_data = pd.read_csv('iomosphere\ionosphere.data')

#Convert the dataset to a NumPy array
ionosphere_np = ionosphere_data.values


