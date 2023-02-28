import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def decision_tree1(data, num_folds):
    X = data.iloc[:, 0:11]
    Y = data.iloc[:, 11]
    kf = KFold(n_splits = num_folds, shuffle = True)
    # c. 
    train_size = []
    test_size = []
    for i, j in list(kf.split(X)):
        train_size.append(len(i))
        test_size.append(len(j))
    print("Kich thuoc cua tap train la: ", set(train_size))
    print("Kich thuoc cua tap test la: ", set(test_size))
    fold = 1
    total_accuracy_score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        # d. xay dung mo hinh cay quyet dinh
        dec_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
        dec_tree.fit(X_train, Y_train)
        Y_pred = dec_tree.predict(X_test)
        """ e. do chinh xac cho tung phan lop cua lan lap cuoi:
                precision    recall   f1-score   support

           4       1.00      0.50      0.67         2
           5       0.67      0.36      0.47        28
           6       0.51      0.89      0.65        46
           7       0.00      0.00      0.00        15
           8       0.00      0.00      0.00         6
        """
        """ f. tinh do chinh xac tong the cho moi lan lap & 
        tinh do chinh xac trung binh """
        print(f'Fold {fold}')
        print("Do chinh xac tong the: ", accuracy_score(Y_test, Y_pred)*100)
        total_accuracy_score += accuracy_score(Y_test, Y_pred)*100
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

def dec_tree2():
    data = {
        "Chieu cao": [180, 167, 136, 174, 141],
        "Do dai mai toc": [15, 42, 35, 15, 28],
        "Giong noi": [0, 1, 1, 0, 1]
    }
    labels = {"Nhan": [0, 1, 1, 0, 1]}

    X = pd.DataFrame(data)
    Y = pd.DataFrame(labels)
    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size = 1/3)
    dec_tree = DecisionTreeClassifier(criterion = "entropy")
    dec_tree.fit(X_train, Y_train)
    print("Ket qua du doan la: ", dec_tree.predict([[135, 39, 1]])) # ket qua du doan la 1

if __name__ == "__main__":
    # Bai tap 1
    # a. doc du lieu
    wine_data = pd.read_csv("winequality-white.csv", sep = ";")
    # b.
    print("Bai tap 1:")
    print("So luong thuoc tinh la: ", len(wine_data.columns)) # so luong thuoc tinh la: 12
    print(np.unique(wine_data["quality"])) # cot nhan la cot "quality", gia tri cua nhan la [3 4 5 6 7 8 9]
    decision_tree1(wine_data, 50)
    # g. so sanh hieu qua giua ba giai thuat KNN, Bayes & Decision Tree
    decision_tree1(wine_data, 60)
    knn(wine_data, 60)
    bayes(wine_data, 60)
    # Bai tap 2
    print("Bai tap 2:")
    dec_tree2()
