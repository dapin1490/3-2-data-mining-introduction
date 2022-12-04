"""
정부에서 제공하는 서울시 초등학교 신체검사 기록인 "student_health_3.csv"을 기반으로 분석하고자 한다.
1. 해당 기록 내에서 수축기, 이완기, 키, 몸무게별 기반으로 학년을 도출하는 KNN, Logistic Regression, Neural Network 모델을 도출하시요. 이때 각 모델별 정확률을 출력하시오.
"""

import pandas as pd
import numpy as np
from sklearn import neighbors  # KNN

global seed

def knn(x, y, num_classes):  # "weeks\week12\sk_3.py" 참고
    knn_model = neighbors.KNeighborsRegressor(num_classes, weights="uniform")  # ["uniform", "distance"]
    knn_model.fit(x)
    y_pred = knn_model.predict(x)
    pass

def logic_reg(x, y):  # "weeks\week13+hw\logicr.py" 참고
    pass

def neu_net(x, y):  # "weeks\week13+hw\verysimplecnn.py" 참고
    pass

data = pd.read_csv(r"weeks\week13+hw\homework\student_health_3.csv", header=0, encoding='euc-kr')
# print(data.info())
# print(data.columns)

seed = 221204  # 랜덤 시드 지정

# 몸무게, 키 : 전학년 다 있음
# 수축기, 이완기 : 1, 4학년만 있음
data_weight = data[["몸무게", "학년"]]
data_height = data[["키", "학년"]]
data_systolic = data[["수축기", "학년"]]
data_diastolic = data[["이완기", "학년"]]
