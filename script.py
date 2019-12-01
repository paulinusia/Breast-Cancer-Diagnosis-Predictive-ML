#for data import and visualization
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def SVM():
    '''
    implement here- or in different file
    f1 regularization since data is imbalanced
    '''
   





if __name__ == '__main__':
    
    '''
    Pre-processing

    data loaded, diagnosis dropped and converted to numerical data
    All 0 data points are replaced with the mean for that column
    Data is standardized

    Dataset is imbalanced.
        1. implement confusion matrix to show how well the model is doing
        2. undersampling (majority) vs oversampling (minority) visualizations
        3. kLearn's resample
        4. SMOTE method 
    
    '''
    
    breast_data = pd.read_csv('./data/data.csv')

    #drop diagnosis
    y = breast_data['diagnosis']
    x = breast_data.drop('diagnosis', axis=1)
    x = x.drop('id', axis=1)
    #x = x.drop('Unnamed: 32', axis=1)


    #replace M and B with 1s and 0s
    y = y.replace(['M', 'B'], [1, 0])
    columns = x.columns


    x = x.replace(0, np.nan)
    #print(x.shape)
    #print(x.columns)

    #print(x.head())
    #print(y.shape)
    #replace missing values with mean
    for col in x.columns:
        x[col].fillna(x[col].mean(), inplace=True)

    
    #print(x)
       

    #standardize the dataset to have a mean of 0, allows us to compare different scales
    scaler = StandardScaler()
    standardized_data = x.copy()

    standardized_data[columns] = pd.DataFrame(scaler.fit_transform(standardized_data[columns]))

    print(standardized_data.head())






