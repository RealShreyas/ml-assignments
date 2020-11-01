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

label_encoder = preprocessing.LabelEncoder()


#One hot encoding the categorical variables-
ohe = pd.get_dummies(occupation['Gender'],prefix='Gender')
occupation = occupation.join(ohe)
del occupation['Gender']


ohe = pd.get_dummies(occupation['Ever_Married'],prefix='Ever_Married')
occupation = occupation.join(ohe)
del occupation['Ever_Married']



ohe = pd.get_dummies(occupation['Profession'],prefix='Profession')
occupation = occupation.join(ohe)
del occupation['Profession']




ohe = pd.get_dummies(occupation['Graduated'],prefix='Graduated')
occupation = occupation.join(ohe)
del occupation['Graduated']



ohe = pd.get_dummies(occupation['Spending_Score'],prefix='Spending_Score')
occupation = occupation.join(ohe)
del occupation['Spending_Score']



ohe = pd.get_dummies(occupation['Var_1'],prefix='Var_1')
occupation = occupation.join(ohe)
del occupation['Var_1']


occupation["Segmentation"] = label_encoder.fit_transform(occupation["Segmentation"])


occupation = occupation[ [ col for col in occupation.columns if col != 'Segmentation' ] + ['Segmentation'] ]


#80-20 split of data into training and test sets.
df_train, df_test = train_test_split(occupation,test_size=0.2)
df2 = df_train.reset_index(drop=True)
df3 = df_test.reset_index(drop=True)
X_train = np.array(df2.iloc[:, :-1])
y_train = df2.iloc[:, -1]
X_test = np.array(df3.iloc[:, :-1])
y_test = df3.iloc[:, -1]


#5-Fold cross validation

folds = StratifiedKFold(n_splits = 5)
accuracy_scores = []
X = np.array(occupation.iloc[:, :-1])
Y = np.array(occupation.iloc[:, -1])

for train_index, test_index in folds.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    #normalise the data
    scaler = StandardScaler()
    # Fit the scalar on training set only.
    scaler.fit(X_train)
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    accuracy_scores.append(return_score(NaiveBayesClassifier(), X_train, y_train, X_test, y_test))

print("The final test accuracy after performing 5 fold cross validation is {} %.".
      format(round(statistics.mean(accuracy_scores)*100,2)))







