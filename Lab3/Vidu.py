import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris

def dec_tree_with_hold_out(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=1/3, random_state=5)
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, Y_train)
    Y_pred = clf_gini.predict(X_test)
    print("Accuracy is: ", accuracy_score(Y_test, Y_pred)*100)
    con_matrix = confusion_matrix(Y_test, Y_pred, labels=[2,0,1])
    print(con_matrix)
    
def dec_tree_with_k_fold(data):
    X = data.data
    Y = data.target
    kf = KFold(n_splits=15, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        clf_gini.fit(X_train, Y_train)
        Y_pred = clf_gini.predict(X_test)
        print("Accuracy is: ", accuracy_score(Y_test, Y_pred)*100)
        con_matrix = confusion_matrix(Y_test, Y_pred)
        print(con_matrix)
        print("==========================")
        
def reg_with_dec_tree(data):
    X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:, 1:5], data.iloc[:, 0], test_size=1/3.0, random_state=100)
    reg = DecisionTreeRegressor(random_state=0)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    err = mean_squared_error(Y_test, Y_pred)
    print(np.sqrt(err))

if __name__ == "__main__":
    iris_data = load_iris()
    housing_data = pd.read_csv("housing_RT.csv")
    reg_with_dec_tree(housing_data)