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
from sklearn.linear_model import Perceptron

#import seaborn as sns

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

#for data import and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from pydotplus import graph_from_dot_data
#from sklearn.tree import export_graphviz
#from pydotplust import graph_from_dot_data

sns.set_style("dark")
colors = ["#800000", "#45ada8", "#2a363b", "#fecea8", "#99b898", "#e5fcc2"]
sns.set_palette(sns.color_palette(colors))

breast_data = pd.read_csv('./data/data.csv')
#breast_data = breast_data.drop(['ID','Unnamed: 32'],axis=1)

#drop diagnosis, create X and Y
y = breast_data['diagnosis']
x_ = breast_data.drop('diagnosis', axis=1)
x = x_.drop('id', axis = 1)

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

def Perceptron():
    grid_param={'tol':[1e-3], "random_state":[0], "max_iter":[15,20,30]}
    perceptron_model= Perceptron()
    perceptron_model.fit(X_train, y_train)
    print("Perceptron")

    return model_eval(mod_select_train(s_run,grid_param))

def KNN():
    grid_param = {'algorithm' : ['brute', 'ball_tree', 'kd_tree'], 'n_neighbors' : [1, 3, 5, 10, 15, 20]}

    knn_run = KNeighborsClassifier()
    knn_run.fit(X_train, y_train)

    print("KNN")

    return model_eval(mod_select_train(knn_run, grid_param))

def SGD():#Adaline SGD

    grid_param = {'penalty' :['l1', 'l2'], 'max_iter':[10, 25, 50, 100]}

    adaline= SGDClassifier()
    adaline.fit(X_train, y_train)

    print('SGD')

    plot.plot(range(1, len(adaline) + 1), adaline, marker='o')
    plot.xlabel('Epochs')
    plot.ylabel('Average Cost')
    plot.tight_layout
    plot.show()

    return model_eval(mod_select_train(adaline, grid_param))



def classifiers():

    best = [KNN(), Perceptron(), SGD()]
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
