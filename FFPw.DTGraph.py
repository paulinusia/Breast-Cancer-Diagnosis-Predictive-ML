# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:00:21 2019

@author: danie
"""

#Preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import  auc
from sklearn import preprocessing


#Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#for data import and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

sns.set_style("dark")
colors = ["#800000", "#45ada8", "#2a363b", "#fecea8", "#99b898", "#e5fcc2"]
sns.set_palette(sns.color_palette(colors))

breast_data = pd.read_csv('data.csv')
#breast_data = breast_data.drop(['ID','Unnamed: 32'],axis=1)

#drop diagnosis, create X and Y
y = breast_data['Diagnosis']
x_ = breast_data.drop('Diagnosis', axis=1)
x = x_.drop('ID', axis = 1)

#replace M and B with 1s and 0s
y = y.replace(['M', 'B'], [1, 0])
columns = x.columns

x = x.replace(0, np.nan)

#replace missing values with mean
for col in x.columns:
    x[col].fillna(x[col].mean(), inplace=True)

#standardize the dataset to have a mean of 0, allows us to compare different scales
scaler = StandardScaler()
standardized_data = x.copy()

standardized_data[columns] = pd.DataFrame(scaler.fit_transform(standardized_data[columns]))

#split the dataset, 70% training, 15% test, 15% development

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state = 0)


#Model Training and Evaluation, Daniella Pombo----------------------------------------------------------

def model_eval(clfr_var): #Evaluates model based on results from test set
    
    prediction = clfr_var.predict(X_test) #Predict class labels for each test sample
    
    cm = confusion_matrix(y_test,prediction) #Returns matrix of confusion matrix values
    acc_s = accuracy_score(y_test, prediction) #Accuracy score of model
    PRE, REC, _ = precision_recall_curve(y_test, prediction, pos_label = 1) #Precision recall curve returns precision, recall and its threshold
    AUC = auc(REC, PRE) #Compute area under precision recall curve
    f_s = f1_score(y_test, prediction) #F1 score gives indication on precision and recall
    cr = classification_report(y_test, prediction) #Returns txt formate of confusion matrix

    return (AUC, f_s, acc_s, cr, clfr_var) #Return tuple of report values
    

def mod_select_train(clfr_var, hypprm): #Train and tone model
    
    #Model Selection
    #Grid search
    gs = GridSearchCV(clfr_var, param_grid = hypprm, cv = 10, scoring = 'f1', refit = True) #Verbose shows u wats going on
    gs.fit(X_train, y_train)
        
    gs = gs.best_estimator_

    # K-fold cross validation
    cross_val_score(estimator = gs, X = X_train, y = y_train, cv = 10, scoring = 'f1')

    return gs

#Run classifiers -------------------------------------------------------------------------

def SupVM(): #Support Vector Machine
    
    grid_param = {'C' :[0.1, 1, 5, 10, 50], 'kernel' :['linear']}
    
    s_run = SVC()
    s_run.fit(X_train, y_train)
    
    print("SVM")
    
    return model_eval(mod_select_train(s_run, grid_param))
    
def lr():#Logistic Regression
    
    grid_param = {'C' :[0.00001, 0.001, 1, 3, 5, 10, 50, 100, 1000]}
    
    lr_run = LogisticRegression()
    lr_run.fit(X_train, y_train)
 
    print("LR")
    
    return model_eval(mod_select_train(lr_run,grid_param))
    

def nnP():#Perceptron Neural Network
    
    grid_param = {'hidden_layer_sizes' : [(100,3), (5,2)], 'max_iter':[25, 50, 100], 'solver': ['adam'], 'activation': ['relu'] }
    
    nnP_run = MLPClassifier()
    nnP_run.fit(X_train, y_train)

    print("NNP")
    
    return model_eval(mod_select_train(nnP_run, grid_param))

def DT(): #Decision Tree
    
    grid_param = {'criterion' : ['gini', 'entropy'], 'max_depth' : [2, 4, 7, 10, 15], 'random_state' : [0]}
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    
    print("DT")

    tree = mod_select_train(tree, grid_param)
    
    #Decision Tree Chart
    #'M', 'B' = 1, 0
    
    dot_data = export_graphviz(tree, out_file = None, impurity = True, filled = True, rounded = True, max_depth = 5, feature_names = ['Radius mean', 'Texture mean', 'Perimeter mean', 'Area mean', 'Smoothness mean', 'Compactness mean', 'Concavity mean', 'Convacve points mean', 'Symmetry mean', 'Fractal dimension', 'Radius se', 'Texture se', 'Perimeter se', 'Area se', 'Smoothness se', 'Compactness se', 'Concavity se', 'Concave points se', 'Symmetry se', 'Fractal dimension se', 'Radius worst', 'Texture worst', 'Perimeter worst', 'Area worst', 'Smoothness worst', 'Compactness worst', 'Concavity worst', 'Concave points worst', 'Symmetry worst', 'Fractal dimension worst'], class_names= ['Begnign', 'Malignant'])
    graph = graph_from_dot_data(dot_data)
    graph.write_png('DecisionTreeGraph.png')
    
    return model_eval(tree)
    

def classifiers():
    
    best = [lr(), DT(), nnP(), SupVM()]
   
    #All classifiers return tuple of (AUC, f_s, acc_s, cr, clfr_var)
    best.sort() #Sort classifiers based on best AUC returned
    
    #Record results/findings (data)
    out_file = open('FinalProjectModelEvaluationReport.txt', 'w')
    
    out_file.writelines('From most significant classifier to least' + '\n')
    
    while len(best) > 0:
        report = best.pop()
        out_file.writelines('Classifier w/ optimal hyperameters: ' + '\n')
        out_file.writelines(str(report[-1])+  '\n')
        out_file.writelines('Confusion matrix'+ '\n')
        out_file.writelines(str(report[-2])+ '\n')
        out_file.writelines('Area under Precision and Recall Curve: ' + str(report[0])+ '\n')
        out_file.writelines('F1s score: ' + str(report[1])+ '\n')
        out_file.writelines('Accuracy: ' + str(report[2]) + '\n')
        out_file.writelines('\n')
        
    out_file.close()
    
classifiers()
