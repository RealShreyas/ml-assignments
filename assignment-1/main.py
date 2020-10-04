import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn import tree

#process input as pandas dataframe
def process_input():
    df = pd.read_csv("AggregatedCountriesCOVIDStats.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') #converting date to a pandas datetime format
    df['Date_month'] = df['Date'].dt.month #extracting month from date
    #df['Date_day'] = df['Date'].dt.day   #Can extract day from date but this doesn't help accuracy
    del df["Date"]      #deleting date column, will keep only month as feature
    #df['Country'] = pd.Categorical(df['Country'])  #use categorical variable for countries
    #df['Country_code'] = df['Country'].cat.codes   #map each country to numerical code
    ohe = pd.get_dummies(df['Country'],prefix='Country')   #use one hot encoding for countries
    df = df.join(ohe)
    del df["Country"]  #deleting country column since we dealt with it via above
    new_deaths = df['Deaths']
    del df['Deaths']
    df = df.join(new_deaths)      #make number of deaths the last column
    df_train, df_test = train_test_split(df,test_size=0.2)   #create random 80-20 split of data
    df2 = df_train.reset_index(drop=True)   #reset indices to start from 0
    df3 = df_test.reset_index(drop=True)
    X_train = df2.iloc[:, :-1]   #input is every column except last
    y_train = df2.iloc[:, -1]    #output is last column
    X_test = df3.iloc[:, :-1]
    y_test = df3.iloc[:, -1]

    return X_train,X_test,y_train,y_test


#Getting best accuracy across 10 random splits
max_accuracy = 0
min_rms_error = 10000
regr_1 = DecisionTreeRegressor(random_state=0)
for i in range(10):
    X_train, X_test, y_train, y_test = process_input()
    regr_1.fit(X_train, y_train)
    y_pred = regr_1.predict(X_test)    #predicted number of deaths on test set
    predicted_output = y_pred.tolist()
    actual_output = y_test.tolist()
    accuracy = r2_score(predicted_output,actual_output)  #using r2 score as accuracy measure
    error = np.sqrt(mean_squared_error(predicted_output,actual_output))
    if accuracy > max_accuracy:
        max_accuracy = accuracy
    if error < min_rms_error:
        min_rms_error = error
print("Max accuracy over 10 random splits is ",max_accuracy)
print("Min rms error over 10 random splits is ",min_rms_error)


#Plotting depth vs accuracy and depth vs rms error graphs
# All possible depth
depths = []
# Best accuracy at the corresponding depth
accuracy = []

#min error at corresponding depth
error = []
min_error = 10000
max_accuracy = 0
depth_of_min_error = 0
depth_of_max_accuracy = 0
#Check what depth is best
for j in range(1, 120):
    current_score_global = 0
    current_error_global = 100000

    # Regression tree with max depth set
    regr_1 = DecisionTreeRegressor(max_depth=j)
    for k in range(10):  #check on 10 random splits for each depth
        X_train, X_test, y_train, y_test = process_input()
        regr_1.fit(X_train, y_train)
        y_pred = regr_1.predict(X_test)   #prediction on test set
        predicted_output = y_pred.tolist()
        actual_output = y_test.tolist()

        # Best score for the current split
        current_score_local = r2_score(actual_output, predicted_output)
        current_error_local = np.sqrt(mean_squared_error(actual_output,predicted_output))
        if current_score_local > current_score_global:
            current_score_global = current_score_local
        if current_error_global > current_error_local:
            current_error_global = current_error_local
        depths.append(j)
        accuracy.append(current_score_global * 100)
        error.append(current_error_global)
    if(current_error_global < min_error):
        min_error = current_error_global
        depth_of_min_error = j
    if(current_score_global > max_accuracy):
        max_accuracy = current_score_global
        depth_of_max_accuracy = j
#print(min_error)
#print(depth_of_min_error)
#print(max_accuracy)
#print(depth_of_max_accuracy)

# Plot depth vs r2 score
plt.plot(depths, accuracy)
# Adds a grid to the plot
plt.grid()
# X-axis label
plt.ylabel('R2 Score')
# Y-axis label
plt.xlabel('Depth')
# Export the plot
plt.savefig('accuracy.png')

# Plot depth vs RMS error
plt.plot(depths, error)
# Adds a grid to the plot
plt.grid()
# X-axis label
plt.ylabel('RMS error')
# Y-axis label
plt.xlabel('Depth')
# Export the plot
plt.savefig('error.png')


