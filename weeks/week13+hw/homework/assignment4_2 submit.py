"""
정부에서 제공하는 서울시 초등학교 신체검사 기록인 "student_health_3.csv"을 기반으로 분석하고자 한다.
1. 해당 기록 내에서 수축기, 이완기, 키, 몸무게별 기반으로 학년을 도출하는 KNN, Logistic Regression, Neural Network 모델을 도출하시요.
이때 각 모델별 정확률을 출력하시오.
"""

import pandas as pd
import tensorflow as tf  # Neural Network
from keras.utils.np_utils import to_categorical
from sklearn import neighbors  # KNN
from sklearn.linear_model import LogisticRegression  # Logistic Regression
# from keras.models import Sequential  # Neural Network
# from keras.layers import Dense, Dropout, BatchNormalization  # Neural Network
from sklearn import metrics  # 정확도 계산
from sklearn.model_selection import train_test_split  # 학습셋 분리

global seed, key


def knn(_key, _data):  # "weeks\week12\sk_3.py" 참고
    x = _data.drop(_key, axis=1).to_numpy()#.reshape(-1, 4)
    y = _data[_key].to_numpy().ravel()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)

    knn_model = neighbors.KNeighborsClassifier(weights="uniform")  # ["uniform", "distance"]
    knn_model.fit(x_tr, y_tr)

    y_pred = knn_model.predict(x_te)
    return metrics.accuracy_score(y_te, y_pred)


def logic_reg(_key, _data):  # "weeks\week13+hw\logicr.py" 참고
    x = _data.drop(_key, axis=1).to_numpy()#.reshape(-1, 4)
    y = _data[_key].to_numpy().ravel()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)

    clf = LogisticRegression(random_state=seed)
    clf.fit(x_tr, y_tr)

    return clf.score(x_te, y_te)


def neu_net(_key, _data, _num_classes, _batch=128, _epoch=50):  # "weeks\week13+hw\verysimplecnn.py" 참고
    num_classes = _num_classes
    batch_size = _batch
    epochs = _epoch

    x = _data.drop(_key, axis=1).to_numpy()
    y = _data[_key].to_numpy().ravel()
    y = to_categorical(y, num_classes)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(x_tr, y_tr, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)
    return model.evaluate(x_te, y_te, verbose=0)[1]


data = pd.read_csv(r"student_health_3.csv", header=0, encoding='euc-kr')
data = data[["몸무게", "키", "수축기", "이완기", "학년"]]
key = "학년"
# print(data.info())

seed = 221204  # 랜덤 시드 지정
tf.random.set_seed(seed)

# 1, 4학년만 있음
# 정수로 구분된 클래스의 숫자가 연속하지 않을 경우 0부터 최댓값까지 원 핫 인코딩하므로 클래스 최댓값 + 1로 사용해야 함

knn_acc = knn(key, data)
logic_reg_acc = logic_reg(key, data)
neu_net_acc = neu_net(key, data, 5, _epoch=43)

print(f"KNN accuracy : {knn_acc :.4f}")
print(f"Logistic Regression accuracy : {logic_reg_acc :.4f}")
print(f"Neural Network accuracy : {neu_net_acc :.4f}")
