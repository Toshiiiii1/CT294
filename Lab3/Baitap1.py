import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def decision_tree(data, num_folds):
    X = data.iloc[:, 0:11]
    Y = data.iloc[:, 11]
    kf = KFold(n_splits = num_folds, shuffle = True)
    fold = 1
    total_accuracy_score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        dec_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
        dec_tree.fit(X_train, Y_train)
        Y_pred = dec_tree.predict(X_test)
        # print(f'Fold {fold}')
        # print("Do chinh xac tong the: ", accuracy_score(Y_test, Y_pred)*100)
        total_accuracy_score += accuracy_score(Y_test, Y_pred)*100
        # print("Do chinh xac cho tung phan lop: ")
        # print(classification_report(Y_test, Y_pred, zero_division = 0))
        # print("==========================")
        fold += 1
    print("Do chinh xac tong the trung binh cua giai thuat Decision Tree la: ", total_accuracy_score/num_folds)
    
def knn(data, num_folds):
    X = data.iloc[:, 0:11]
    Y = data.iloc[:, 11]
    kf = KFold(n_splits = num_folds, shuffle = True)
    total_accuracy_score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        knn_model = KNeighborsClassifier(n_neighbors = 5)
        knn_model.fit(X_train, Y_train)
        Y_pred = knn_model.predict(X_test)
        total_accuracy_score += accuracy_score(Y_test, Y_pred)*100
    print("Do chinh xac tong the trung binh cua giai thuat KNN la: ", total_accuracy_score/num_folds)
    
def bayes(data, num_folds):
    X = data.iloc[:, 0:11]
    Y = data.iloc[:, 11]
    kf = KFold(n_splits = num_folds, shuffle = True)
    total_accuracy_score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        bayes_model = GaussianNB()
        bayes_model.fit(X_train, Y_train)
        Y_pred = bayes_model.predict(X_test)
        total_accuracy_score += accuracy_score(Y_test, Y_pred)*100
    print("Do chinh xac tong the trung binh cua giai thuat Bayes la: ", total_accuracy_score/num_folds)

if __name__ == "__main__":
    # a. doc du lieu
    wine_data = pd.read_csv("winequality-white.csv", sep = ";")
    # b.
    # print(len(wine_data.columns)) # so luong thuoc tinh la: 12
    # print(np.unique(wine_data["quality"])) # cot nhan la cot "quality", gia tri cua nhan la [3 4 5 6 7 8 9]
    decision_tree(wine_data, 50)
    decision_tree(wine_data, 60)
    knn(wine_data, 60)
    bayes(wine_data, 60)