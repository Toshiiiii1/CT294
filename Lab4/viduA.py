import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# ??????????????????
def LR1(X, Y, eta, n, theta0, theta1):
    m = len(X)
    for k in range(0, n):
        # print("Lan lap: ", k)
        for i in range(0, m):
            h_i = theta0 + theta1 * X[i]
            theta0 = theta0 + eta * (Y[i] - h_i) *1
            # print("Phan tu ", i, "y = ", Y[i], "h = ", h_i, "gia tri theta0 = ", round(theta0, 3))
            theta1 = theta1 + eta * (Y[i] - h_i) * X[i]
            # print("Phan tu ", i, "gia tri theta1 = ", round(theta1, 3))
    return [round(theta0, 3), round(theta1, 3)]

if __name__ == "__main__":
    X = np.array([1,2,4])
    Y = np.array([2,3,6])
    # a. bieu dien du lieu len mat phang toa do
    plt.axis([0,5,0,8])
    plt.plot(X, Y, "ro", color = "blue")
    plt.xlabel("Gia tri thuoc tinh X")
    plt.ylabel("Gia tri thuoc tinh Y")
    # plt.show()
    
    # b. tim ham hoi qui
    # theta = LR1(X, Y, 0.2, 1, 0, 1)
    # print(theta)
    
    # c. ve duong hoi qui
    theta = LR1(X, Y, 0.1, 1, 0, 1)
    X1 = np.array([1,6])
    Y1 = theta[0] + theta[1] * X1

    theta2 = LR1(X, Y, 0.1, 2, 0, 1)
    X2 = np.array([1,6])
    Y2 = theta2[0] + theta2[1] * X2
    
    plt.axis([0,7,0,10])
    plt.plot(X, Y, "ro", color = "blue")
    
    plt.plot(X1, Y1, color = "violet")
    plt.plot(X2, Y2, color = "green")
    
    plt.xlabel("Gia tri thuoc tinh X")
    plt.ylabel("Gia tri thuoc tinh Y")
    plt.show()
    
    # data = pd.read_csv("Housing_2019.csv", sep = ",", index_col=0)
    # # print(data.to_string())
    # X = data.iloc[:, [1,2,3,4,10]]
    # Y = data.price
    # # print(X.shape)
    # # plt.scatter(data.lotsize, data.price)
    # # plt.show()
    # lm = linear_model.LinearRegression()
    # lm.fit(X.iloc[0:520], Y.iloc[0:520])
    # print(lm.intercept_)
    # print(lm.coef_)
    # X_test = X.iloc[-20:]
    # Y_test = Y.iloc[-20:]
    # Y_pred = lm.predict(X_test)
    # err = mean_squared_error(Y_test, Y_pred)
    # rmse_err = np.sqrt(err)
    # print(round(rmse_err, 3))