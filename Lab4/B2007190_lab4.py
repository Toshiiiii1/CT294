import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def LR1(X, Y, eta, n, theta0, theta1):
    m = len(X)
    for k in range(0, n):
        for i in range(0, m):
            h_i = theta0 + theta1 * X[i]
            theta0 = theta0 + eta * (Y[i] - h_i) *1
            theta1 = theta1 + eta * (Y[i] - h_i) * X[i]
    return [round(theta0, 3), round(theta1, 3)]

def ex1():
    X = np.array([150, 147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
    Y = np.array([90, 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])
    
    # bieu dien tap du lieu len mat phang Oxy
    plt.axis([0, 200, 0, 100])
    plt.plot(X, Y, "ro")
    plt.xlabel("Gia tri X")
    plt.ylabel("Gia tri Y")
    # plt.show()
    
    # ve duong hoi quy len mat phang Oxy
    theta = LR1(X, Y, 0.00001, 3, 0, 1)
    X1 = np.array([147, 183])
    Y1 = theta[0] + theta[1] * X1
    plt.plot(X1, Y1, color = "violet")
    print(theta)
    print(Y1)
    plt.show()
    
    # nhan xet ket qua
    
    
    # cap nhat code
    
    
def ex2():
    # su dung chi so MSE va chi so RMSE de danh gia mo hinh hoi quy tuyen tinh
    data = pd.read_csv("Housing_2019.csv", sep = ",", index_col=0)
    X = data.loc[:, ["lotsize", "bedrooms", "stories", "garagepl"]]
    Y = data.loc[:, "price"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3.0, random_state = 100)
    lm = linear_model.LinearRegression()
    lm.fit(X_train, Y_train)
    Y_pred = lm.predict(X_test)
    mse_score = mean_squared_error(Y_test, Y_pred)
    rmse_score = np.sqrt(mse_score)
    print(round(mse_score, 3), round(rmse_score, 3))
    
def ex3():
    X = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]) # ham luong go cung
    Y = np.array([6.3, 11.1, 20.0, 24.0, 26.1, 30.0, 33.8, 34.0, 38.1, 39.9, 42.0, 46.1, 53.1, 52.0, 52.5, 48.0, 42.8, 27.8, 21.9]) # do cang manh
    
    # xay dung bieu do the hien moi lien he giua X va Y
    plt.axis([0, 20, 0, 60])
    plt.plot(X, Y, "bo")
    plt.xlabel("Ham luong go cung")
    plt.ylabel("Do cang manh")
    # plt.show()
    
    # xay dung phuong trinh
    mymodel = np.poly1d(np.polyfit(X, Y, 3))
    myline = np.linspace(1, 20, 100)
    plt.plot(myline, mymodel(myline))
    plt.show()
    print(mymodel)
    
def ex4(X, Y):
    # su dung giai thuat Rung ngau nhien de du doan chat luong ruou vang voi nghi thuc hold-out
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3.0, random_state = 100)
    random_forget = RandomForestClassifier(n_estimators = 50)
    random_forget.fit(X_train, Y_train)
    Y_pred = random_forget.predict(X_test)
    matrix = confusion_matrix(Y_test, Y_pred)
    print("Ma tran nham lan: ")
    print(matrix)
    print("Do chinh xac: ", accuracy_score(Y_test, Y_pred))
    
def ex5(X, Y, num_folds):
    # su dung giai thuat "AdaBoostClassifier" de du doan chat luong ruou vang voi nghi thuc K-Fold
    kf = KFold(n_splits = num_folds, shuffle = True)
    total_accuracy_score = 0
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        abc = AdaBoostClassifier(n_estimators = 50)
        abc.fit(X_train, Y_train)
        Y_pred = abc.predict(X_test)
        total_accuracy_score += accuracy_score(Y_test, Y_pred)*100
    print(total_accuracy_score / num_folds)
        
            
if __name__ == "__main__":
    # Bai tap 1
    ex1()
    # Bai tap 2
    ex2()
    # Bai tap 3
    ex3()
    # tao du lieu cho bai tap 4 & bai tap 5
    wine = load_wine()
    wine_data = pd.DataFrame(data= np.c_[wine['data'], wine['target']], columns= wine['feature_names'] + ['target'])
    X = wine_data.iloc[:, 0:-1]
    Y = wine_data.target
    # Bai tap 4
    ex4(X, Y)
    # Bai tap 5
    ex5(X, Y, 50)