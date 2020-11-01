import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import statistics
from sklearn.decomposition import PCA
from model import NaiveBayesClassifier
from model import return_score


#Handling missing attributes
occupation = pd.read_csv("Train_A.csv").dropna()
occupation = occupation.reset_index(drop='True')

#ID does not contribute to predicting segmentation
del occupation['ID']

head = list(occupation.columns)

label_encoder = preprocessing.LabelEncoder()


occupation["Gender"] = label_encoder.fit_transform(occupation["Gender"])


occupation["Ever_Married"] = label_encoder.fit_transform(occupation["Ever_Married"])


occupation["Profession"] = label_encoder.fit_transform(occupation["Profession"])


occupation["Graduated"] = label_encoder.fit_transform(occupation["Graduated"])

occupation["Spending_Score"] = label_encoder.fit_transform(occupation["Spending_Score"])

occupation["Var_1"] = label_encoder.fit_transform(occupation["Var_1"])

occupation["Segmentation"] = label_encoder.fit_transform(occupation["Segmentation"])


occupation = occupation[ [ col for col in occupation.columns if col != 'Segmentation' ] + ['Segmentation'] ]

folds = StratifiedKFold(n_splits = 5)

"""
We are going to remove the outliers or feature values in sample that are greater than means + 3 * standard deviation.
We will first convert it to numpy array for ease of operations and convert it back to pandas dataframe
"""

# Calculate the mean for each feature
means = np.array(occupation.mean(axis=0))

#Calulate the standard deviation for each feature
sd = np.array(occupation.std(axis=0))

occupations = occupation.to_numpy()

occupation = pd.DataFrame(occupations)

"""
We perform sequential back ward selection and reduce the feature set.
After this we print the features and then proceed to print the accuracy
"""

SBS = []
accuracy = 0
df_train, df_test = train_test_split(occupation,test_size=0.2)
df2 = df_train.reset_index(drop=True)
df3 = df_test.reset_index(drop=True)
X_train = np.array(df2.iloc[:, :-1])
y_train = df2.iloc[:, -1]
X_test = np.array(df3.iloc[:, :-1])
y_test = df3.iloc[:, -1] 
new_accuracy= return_score(NaiveBayesClassifier(), X_train, y_train, X_test, y_test)


while True:
    accuracy_scores = []
    
    for column in range(len(occupation.columns)-1):
        occupation_copy = occupation.copy(deep=True)
        occupation_copy.drop(column,axis=1)

        df_train, df_test = train_test_split(occupation_copy, test_size = 0.2)
        df2 = df_train.reset_index(drop=True)
        df3 = df_test.reset_index(drop=True)
        X_train = np.array(df2.iloc[:, :-1])
        y_train = df2.iloc[:, -1]
        X_test = np.array(df3.iloc[:, :-1])
        y_test = df3.iloc[:, -1]
        accuracy_scores.append(return_score(NaiveBayesClassifier(), X_train, y_train, X_test, y_test))
    
    column_to_delete = np.argmax(accuracy_scores)
    
    if(accuracy_scores[column_to_delete] > accuracy):
        SBS.append(column_to_delete)
        occupation.drop(column_to_delete,axis=1)
        accuracy = accuracy_scores[column_to_delete]
    else:
        break

index = 0
features = []
SBS.sort()

for i in range(len(head)):
    if index < int(len(SBS)):
        if i == SBS[index]:
            index = index + 1
        else:
            features.append(head[i])
    else:
        pass


print("The features selected after sequential backward selection is as follows: {}".format(features))

# Perform k fold cross validation on new set of features
occupation_sbs = occupation.drop(SBS, axis=1)

acc_scores = []
X = np.array(occupation_sbs.iloc[:, :-1])
Y = np.array(occupation_sbs.iloc[:, -1])

for train_index, test_index in folds.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    acc_scores.append(return_score(NaiveBayesClassifier(), X_train, y_train, X_test, y_test))

print("The final test accuracy after performing 5 fold cross validation and sequential backward selection is {} %.".format(round(statistics.mean(acc_scores)*100,2)))