import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def decision_tree(X_train, X_test, Y_train, Y_test):
    tree = DecisionTreeRegressor(random_state=0)
    bagging_regtree = BaggingRegressor(estimator=tree, n_estimators=10, random_state=42)
    bagging_regtree.fit(X_train, Y_train)
    Y_pred = bagging_regtree.predict(X_test)
    err = mean_squared_error(Y_test, Y_pred)
    print(np.sqrt(err))

def linear_reg(X_train, X_test, Y_train, Y_test):
    lm = linear_model.LinearRegression()
    bagging_reg = BaggingRegressor(estimator=lm, n_estimators=10, random_state=42)
    bagging_reg.fit(X_train, Y_train)
    Y_pred = bagging_reg.predict(X_test)
    err = mean_squared_error(Y_test, Y_pred)
    print(np.sqrt(err))

if __name__ == "__main__":
    data = pd.read_csv("Housing_2019.csv", sep = ",", index_col=0)
    X = data.iloc[:, [1,2,4,10]]
    Y = data.price
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3.0, random_state = 100)
    decision_tree(X_train, X_test, Y_train, Y_test)
    linear_reg(X_train, X_test, Y_train, Y_test)