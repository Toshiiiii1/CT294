import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

""" đọc dữ liệu từ file csv bằng pandas """
iris_data = pd.read_csv("iris_data.csv")
att = iris_data.iloc[:,0:4]
label = iris_data.nhan

""" 
    - phân chia dữ liệu theo cách Hold-out với 2/3 dữ liệu dùng
    làm để huấn luyện mô hình học, 1/3 dữ liệu dùng để kiểm tra mô hình
    - X_train, Y_train: tập dữ liệu và nhãn dùng để huấn luyện
    - X_test: tập dữ liệu dùng để kiểm tra mô hình, kết quả trả về là kết quả trên lý thuyết
    - Y_test: tập kết quả trên thực tế
"""
X_train, X_test, Y_train, Y_test = train_test_split(att, label, test_size = 1/3.0, random_state = 5)

""" xây dựng mô hình KNN """
knn_model = KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(X_train, Y_train)

""" xây dựng mô hình Bayes """
bayes_model = GaussianNB()
bayes_model.fit(X_train, Y_train)

""" dự đoán nhãn của tập dữ liệu X_test và tính toán độ chính xác của KNN """
y_pred = knn_model.predict(X_test)
print("Accuracy is: ", accuracy_score(Y_test, y_pred)*100)
print(confusion_matrix(Y_test, y_pred, labels = ["Iris-virginica", "Iris-setosa", "Iris-versicolor"]))

""" dự đoán nhãn của tập dữ liệu X_test và tính toán độ chính xác của Bayes """
y_pred = bayes_model.predict(X_test)
print("Accuracy is: ", accuracy_score(Y_test, y_pred)*100)
print(confusion_matrix(Y_test, y_pred))