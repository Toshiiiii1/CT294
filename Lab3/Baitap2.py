import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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
print(dec_tree.predict([[135, 39, 1]])) # ket qua du doan la 1