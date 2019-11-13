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

def preprocess():
    #load data
    breast_data = pd.read_csv('./data/data.csv')
    



    
    #drop diagnosis
    y = breast_data['diagnosis']
    x = breast_data.drop('diagnosis', axis=1)
    
    #replace M and B with 1s and 0s
    y = y.replace(['M', 'B'], [1, 0])

    columns = x.columns

    #standardize the dataset to have a mean of 0, allows us to compare different scales
    scaler = StandardScaler()

    standardized_data = x.copy()
    standardized_data[columns] = pd.DataFrame(scaler.fit_transform(standardized_data[columns]))


    for col in x.columns:
        #replace missing values with nAn
        x[col].replace(0, np.NaN)
        #print(x[col])
        #replace nAn with mean of each column
        x[col].fillna(x[col].mean(), inplace=True)
        print(x[col])

   





if __name__ == '__main__':
    preprocess()








