import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

def process_input():
    df = pd.read_csv("AggregatedCountriesCOVIDStats.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') #converting date to a pandas datetime format
    df['Date_month'] = df['Date'].dt.month #extracting month from date
    df['Date_day'] = df['Date'].dt.day
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]     #rearranging columns of dataframe
    df = df[cols]
    del df["Date"]      #deleting date column, will keep only month (can additionally keep days also?)
    ohe = pd.get_dummies(df['Country'],prefix='Country')
    df = df.join(ohe)
    del df["Country"]
    new_deaths = df['Deaths']
    del df['Deaths']
    df = df.join(new_deaths)
    #print(df.head())
    #
    #cols = df.columns.tolist()
    #cols = cols[-1:] + cols[:-1]  # rearranging columns of dataframe
    #df = df[cols]
    #print(df.columns)
    # print(df.var())
    X = df.iloc[:, :-1].values    #defining input as all but last attribute
    # print(X)
    y = df.iloc[:, -1].values     #output is "No. of deaths" (last attribute)
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  #Random 80-20 split into train and test
    #Since pandas dataframes are easier to deal with than ndarrays, we will convert back to dataframe
    X_train = pd.DataFrame(X_train)
    #X_train.columns = ["Country","Date_month","Confirmed","Recovered"]
    X_test = pd.DataFrame(X_test)
    #X_test.columns = ["Country","Date_month","Confirmed","Recovered"]
    y_train = pd.DataFrame(y_train)
    y_train.columns = ["Deaths"]
    y_test = pd.DataFrame(y_test)
    y_test.columns = ["Deaths"]
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

# All possible depth
depths = []
# Best accuracy at the corresponding depth
accuracy = []

for j in range(90,120):
    #min_error = 10000
    # Best score over 10 splits
    current_score_global = 0
    # Regression tree with max depth set
    regr_1 = DecisionTreeRegressor(max_depth=j)
    for i in range(10):
        X_train, X_test, y_train, y_test = process_input()
        regr_1.fit(X_train, y_train)
        predictions = regr_1.predict(X_train)

        y_pred = regr_1.predict(X_test)
        predicted_output = y_pred.tolist()
        actual_output = y_test['Deaths'].tolist()
        
        # Best score for the current split
        current_score_local = r2_score(actual_output, predicted_output)
        if current_score_local > current_score_global:
            current_score_global = current_score_local
        
        depths.append(j)
        accuracy.append(current_score_global*100)

# Adds the line in the graph
plt.plot(depths, accuracy)
# Adds a grid to the plot
plt.grid() 
# X-axis label
plt.ylabel('Accuracy') 
# Y-axis label
plt.xlabel('Depth')
# Export the plot
plt.savefig('accuracy.png')