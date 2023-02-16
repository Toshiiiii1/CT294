import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

# a. đọc dữ liệu rượu vang đỏ
data = pd.read_csv("winequality-red.csv", sep = ";")

# b.
len(data) # tập dữ liệu có 1599 phần tử
np.unique(data["quality"]).size # tập dữ liệu có 6 nhãn [3,4,5,6,7,8]

# c. 
att = data.iloc[:,0:10] # tập các thuộc tính
labels = data["quality"] # tập các nhãn
""" 
    - X_train, Y_train: tập dữ liệu và nhãn dùng để huấn luyện
    - X_test: tập dữ liệu dùng để kiểm tra mô hình, tổng số phần
    tử trong tập kiểm tra là 640, kết quả trả về là nhãn của các phần tử trong
    tập X_test trên lý thuyết
    - Y_test: tập nhãn của các phần tử trong X_test trên thực tế
"""
X_train, X_test, Y_train, Y_test = train_test_split(att, labels, test_size = 0.4)

# d. Xây dựng mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors = 7)
knn_model.fit(X_train, Y_train)

# i. Sử dụng tập X_test để kiểm tra mô hình KNN
Y_pred = knn_model.predict(X_test)
print("Do chinh xac tong the cua mo hinh KNN:", accuracy_score(Y_test, Y_pred)*100)
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
print("Do chinh xac cua tung lop trong mo hinh KNN:")
print(matrix.diagonal()/matrix.sum(axis=1))

# ii. Sử dụng 8 phần tử đầu tiên của tập X_test để kiểm tra mô hình KNN
X_test2 = X_test.iloc[0:8]
Y_test2 = Y_test.iloc[0:8]
Y_pred = knn_model.predict(X_test2)
print("Do chinh xac tong the cua mo hinh KNN voi 8 phan tu:", accuracy_score(Y_test2, Y_pred)*100)
matrix = confusion_matrix(Y_test2, Y_pred)
print(matrix)
print("Do chinh xac cua tung lop trong mo hinh KNN voi 8 phan tu:")
print(matrix.diagonal()/matrix.sum(axis=1))

# e. Xây dựng mô hình Bayes
bayes_model = GaussianNB()
bayes_model.fit(X_train, Y_train)
Y_pred = bayes_model.predict(X_test)
print("Do chinh xac tong the cua mo hinh Bayes voi 8 phan tu:", accuracy_score(Y_test, Y_pred)*100)
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
print("Do chinh xac cua tung lop trong mo hinh Bayes voi 8 phan tu:")
print(matrix.diagonal()/matrix.sum(axis=1))

# f. Chia tập dữ liệu với nghi thức hold-out & so sánh độ chính xác tổng thể giữa KNN & Bayes
X_train, X_test, Y_train, Y_test = train_test_split(att, labels, test_size = 1/3)
# knn_model = KNeighborsClassifier(n_neighbors = 7)
knn_model.fit(X_train, Y_train) # train lại mô hình KNN
# bayes_model = GaussianNB()
bayes_model.fit(X_train, Y_train) # train lại mô hình Bayes
# đưa ra dự đoàn & tính độ chính xác tổng thể cho mô hình KNN
Y_pred = knn_model.predict(X_test)
accuracy_score_of_knn = accuracy_score(Y_test, Y_pred)*100
# đưa ra dự đoàn & tính độ chính xác tổng thể cho mô hình Bayes
Y_pred = bayes_model.predict(X_test)
accuracy_score_of_bayes = accuracy_score(Y_test, Y_pred)*100
print(accuracy_score_of_knn, accuracy_score_of_bayes)

""" Nhận xét: độ chính xác giữa hai thuật toán học không chênh lệch quá cao, độ chính 
xác tổng thể của KNN cao hơn hoặc ngược lại tùy theo tập dữ liệu được chia """