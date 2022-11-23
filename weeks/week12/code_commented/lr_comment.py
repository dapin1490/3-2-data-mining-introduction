# Code source: Jaques Grobler
# License: BSD 3 clause


# 0. 패키지 로드
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# 1. 데이터셋 로드
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# print(diabetes_X.shape)
# print(diabetes_y.shape)


# 2. 하나의 특징만 추출
diabetes_X = diabetes_X[:, np.newaxis, 2]
# print(diabetes_X.shape)

# 3. 입력값에 대한 훈련/테스트 데이터셋 분리
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 4. 출력값에 대한 훈련/테스트 데이터셋 분리
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 5. 선형 회귀에 대한 모델 로드
regr = LinearRegression()

# 6. 선형 회귀 훈련
regr.fit(diabetes_X_train, diabetes_y_train)

# 7. 테스트 입력값으로 예측하기
diabetes_y_pred = regr.predict(diabetes_X_test)

# 8. 기울기 값 출력
print("Coefficients: \n", regr.coef_)

# 9.MSE 계산
print("Mean squared error: %.2f" %
      mean_squared_error(diabetes_y_test / np.mean(diabetes_y_test),
                         diabetes_y_pred / np.mean(diabetes_y_pred)))
# The coefficient of determination: 1 is perfect prediction

# 10. R^2 값 계산
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# 11. 결과값 산포도와 라인 겹쳐서 출력하기
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
# plt.xticks(())
# plt.yticks(())
plt.show()
