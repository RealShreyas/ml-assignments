import numpy as np
import pandas as pd
import statistics
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
from numpy.random import randint
from prettytable import PrettyTable

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
spam = pd.read_csv(url, header=None).dropna()
scaler = MinMaxScaler()
scaler.fit(spam)
spam = pd.DataFrame(scaler.transform(spam))

df_train, df_test = train_test_split(spam,test_size=0.2)
df2 = df_train.reset_index(drop=True)
df3 = df_test.reset_index(drop=True)
X_train = np.array(df2.iloc[:, :-1])
y_train = df2.iloc[:, -1]
X_test = np.array(df3.iloc[:, :-1])
y_test = df3.iloc[:, -1]

clf1 = svm.SVC(kernel='linear', C = 1)
clf1.fit(X_train,y_train)
output = clf1.predict(X_test)
accuracy = accuracy_score(y_test.tolist(), output.tolist())
print("The accuracy of SVM with linear kernel is {} %.".format(round(accuracy*100,2)))

clf2 = svm.SVC(kernel='poly', degree=2, C = 1)
clf2.fit(X_train,y_train)
output = clf2.predict(X_test)
accuracy = accuracy_score(y_test.tolist(), output.tolist())
print("The accuracy of SVM with quadratic kernel is {} %.".format(round(accuracy*100,2)))

clf3 = svm.SVC(kernel='rbf', C = 1)
clf3.fit(X_train,y_train)
output = clf3.predict(X_test)
accuracy = accuracy_score(y_test.tolist(), output.tolist())
print("The accuracy of SVM with radial basis function kernel is {} %.".format(round(accuracy*100,2)))

# seed random number generator
seed(1)
# generate some integers
c_values = randint(1, 100, 10)

def return_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    output = model.predict(X_test)
    accuracy = accuracy_score(y_test.tolist(), output.tolist())
    return accuracy

folds = StratifiedKFold(n_splits = 5)

df_train, df_test = train_test_split(spam,test_size=0.2)
df2 = df_train.reset_index(drop=True)
df3 = df_test.reset_index(drop=True)
X = np.array(df2.iloc[:, :-1])
Y = df2.iloc[:, -1]
X_test = np.array(df3.iloc[:, :-1])
y_test = df3.iloc[:, -1] 

linear_train_accuracy_scores = []
linear_test_accuracy_scores = []

for c in c_values:
    train_accuracy = []
    for train_index, test_index in folds.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        train_accuracy.append(return_score(svm.SVC(kernel="linear", C=c), X_train, y_train, X_test, y_test))
    
    linear_train_accuracy_scores.append(round(statistics.mean(train_accuracy)*100,2))
    
    test_accuracy = return_score(svm.SVC(kernel="linear", C=c), X, Y, X_test, y_test)
                                 
    linear_test_accuracy_scores.append(round(test_accuracy*100,2))

p = PrettyTable()
x = np.column_stack((c_values, linear_train_accuracy_scores,linear_test_accuracy_scores))

print("The train and test accuracies for differnt C values for SVM with linear kernel is as follows:")

p.field_names = ["C value", "Train Accuracy", "Test Accuracy"]
for row in x:
    p.add_row(row)

print(p)

quadratic_train_accuracy_scores = []
quadratic_test_accuracy_scores = []

for c in c_values:
    train_accuracy = []
    for train_index, test_index in folds.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        train_accuracy.append(return_score(svm.SVC(kernel='poly', degree=2, C=c), X_train, y_train, X_test, y_test))
    
    quadratic_train_accuracy_scores.append(round(statistics.mean(train_accuracy)*100,2))
    
    test_accuracy = return_score(svm.SVC(kernel='poly', degree=2, C=c), X, Y, X_test, y_test)
    
    quadratic_test_accuracy_scores.append(round(test_accuracy*100,2))

q = PrettyTable()
y = np.column_stack((c_values, quadratic_train_accuracy_scores,quadratic_test_accuracy_scores))

print("The train and test accuracies for differnt C values for SVM with quadratic kernel is as follows:")

q.field_names = ["C value", "Train Accuracy", "Test Accuracy"]
for row in y:
    q.add_row(row)

print(q)

rbf_train_accuracy_scores = []
rbf_test_accuracy_scores = []

for c in c_values:
    train_accuracy = []
    for train_index, test_index in folds.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        train_accuracy.append(return_score(svm.SVC(kernel='rbf', C=c), X_train, y_train, X_test, y_test))
    
    rbf_train_accuracy_scores.append(round(statistics.mean(train_accuracy)*100,2))
    
    test_accuracy = return_score(svm.SVC(kernel='rbf', C=c), X, Y, X_test, y_test)
    
    rbf_test_accuracy_scores.append(round(test_accuracy*100,2))

r = PrettyTable()
z = np.column_stack((c_values, rbf_train_accuracy_scores,rbf_test_accuracy_scores))

print("The train and test accuracies for differnt C values for SVM with radial basis function kernel is as follows:")

r.field_names = ["C value", "Train Accuracy", "Test Accuracy"]
for row in z:
    r.add_row(row)

print(r)