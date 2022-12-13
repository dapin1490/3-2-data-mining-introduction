import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

x = pd.read_csv("student_health_3.csv", encoding="CP949")

year = x["학년"]
input_data = np.array([x["키"], x["몸무게"], x["수축기"], x["이완기"]]).T
n_neighbors = 5

X_train, X_test, y_train, y_test = train_test_split(input_data, year, random_state=42)

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

logicr = LogisticRegression()
logicr.fit(X_train, y_train)
print(logicr.score(X_test, y_test))


nn = MLPClassifier()
nn.fit(X_train, y_train)
print(nn.score(X_test, y_test))