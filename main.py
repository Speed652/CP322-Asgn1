import pandas as pd
import numpy as np
from functions import LogisticRegression
from functions import KNearestNeighbor
from functions import clean_data


ionosphere_data = pd.read_csv('ionosphere/ionosphere.data', encoding='latin1')
adult_data = pd.read_csv('adult/adult.data', encoding='latin1')
heart_disease_data = pd.read_csv('heart_disease/cleveland.data', encoding='latin1')
iris_data = pd.read_csv('iris/iris.data', encoding='latin1')

#Convert the dataset to a NumPy array
ionosphere_np = ionosphere_data.values
adult_data_np = adult_data.values
heart_disease_np = heart_disease_data.values
iris_np = iris_data.values



ionosphere_X = ionosphere_np[:, :-1]
ionosphere_y = ionosphere_np[:, -1]

adult_X = adult_data_np[:, :-1]
adult_y = adult_data_np[:, -1]

heart_disease_X = heart_disease_np[:, :-1]
heart_disease_y = heart_disease_np[:, -1]

iris_X = iris_np[:, :-1]
iris_y = iris_np[:, -1]

# Initialize logistic regression model
lr_model = LogisticRegression()

# Perform 5-fold cross-validation for logistic regression on ionosphere dataset
ionosphere_X_cleaned = clean_data(ionosphere_X)
average_accuracy_lr_ionosphere = lr_model.cross_validate(ionosphere_X_cleaned, ionosphere_y)
print(f'Average Accuracy for Logistic Regression on Ionosphere dataset: {average_accuracy_lr_ionosphere:.4f}')

# Perform 5-fold cross-validation for logistic regression on adult dataset
adult_X_cleaned = clean_data(adult_X)
average_accuracy_lr_adult = lr_model.cross_validate(adult_X_cleaned, adult_y)
print(f'Average Accuracy for Logistic Regression on Adult dataset: {average_accuracy_lr_adult:.4f}')

# Perform 5-fold cross-validation for logistic regression on heart disease dataset
heart_disease_X_cleaned = clean_data(heart_disease_X)
average_accuracy_lr_heart = lr_model.cross_validate(heart_disease_X_cleaned, heart_disease_y)
print(f'Average Accuracy for Logistic Regression on Heart Disease dataset: {average_accuracy_lr_heart:.4f}')

# Perform 5-fold cross-validation for logistic regression on iris dataset
iris_X_cleaned = clean_data(iris_X)
average_accuracy_lr_iris = lr_model.cross_validate(iris_X_cleaned, iris_y)
print(f'Average Accuracy for Logistic Regression on Iris dataset: {average_accuracy_lr_iris:.4f}')

# Initialize k-Nearest Neighbor model
knn_model = KNearestNeighbor()

# Perform 5-fold cross-validation for k-Nearest Neighbor on ionosphere dataset
average_accuracy_knn_ionosphere = knn_model.cross_validate(ionosphere_X_cleaned, ionosphere_y)
print(f'Average Accuracy for k-Nearest Neighbor on the Ionosphere dataset: {average_accuracy_knn_ionosphere:.4f}')

# Perform 5-fold cross-validation for k-Nearest Neighbor on the adult dataset
average_accuracy_knn_adult = knn_model.cross_validate(adult_X_cleaned, adult_y)
print(f'Average Accuracy for k-Nearest Neighbor on the Adult dataset: {average_accuracy_knn_adult:.4f}')

# Perform 5-fold cross-validation for k-Nearest Neighbor on the heart disease dataset
average_accuracy_knn_heart = knn_model.cross_validate(heart_disease_X_cleaned, heart_disease_y)
print(f'Average Accuracy for k-Nearest Neighbor on the Heart Disease dataset: {average_accuracy_knn_heart:.4f}')

# Perform 5-fold cross-validation for k-Nearest Neighbor on the iris dataset
average_accuracy_knn_iris = knn_model.cross_validate(iris_X_cleaned, iris_y)
print(f'Average Accuracy for k-Nearest Neighbor on the Iris dataset: {average_accuracy_knn_iris:.4f}')




