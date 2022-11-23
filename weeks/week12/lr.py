# Code source: Jaques Grobler
# License: BSD 3 clause

# 0. 패키지 로드
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import time

# 참고 : https://wikidocs.net/26
res = open("res/lr_result.txt", 'w')

res.write(f"lr.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}\n\n")
print(f"lr.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}")

res.write("비만 데이터셋으로 선형 회귀\n\n")

# Load the diabetes dataset
# 1. 데이터셋 로드
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

res.write(f"type(diabetes_X) : {type(diabetes_X)}\n")
res.write(f"diabetes_X.shape : {diabetes_X.shape}\n")
res.write(f"diabetes_y.shape : {diabetes_y.shape}\n")

# Use only one feature
# 2. 하나의 특징만 추출
diabetes_X = diabetes_X[:, np.newaxis, 2]
res.write(f"특징 추출 이후 diabetes_X.shape : {diabetes_X.shape}\n")

# Split the data into training/testing sets
# 3. 학습셋 테스트셋 분리
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
# 4. 출력값에 대한 학습셋 테스트셋 분리
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
# 5. 선형 회귀 모델 로드
regr = linear_model.LinearRegression()

# Train the model using the training sets
# 6. 선형 회귀 훈련
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
# 7. 테스트셋으로 테스트하기
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
# 8. 기울기 값 출력
print("Coefficients: \n", regr.coef_)
# The mean squared error
# 9. MSE 계산 : 0에 가까울수록 좋음
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
# 10. r2_score 계산 : 1에 가까울수록 좋음
# 근데 이게 0.47이 나와? 썩 별론데
print("Coefficient of determination(r2_score): %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

res.write(f"\nCoefficients: {regr.coef_}\n")
res.write(f"Mean squared error 1: {mean_squared_error(diabetes_y_test, diabetes_y_pred) :.2f}\n")
res.write(f"Mean squared error 2: {mean_squared_error(diabetes_y_test/np.mean(diabetes_y_test), diabetes_y_pred/np.mean(diabetes_y_pred)) :.2f}\n")
res.write(f"Coefficient of determination(r2_score): {r2_score(diabetes_y_test, diabetes_y_pred) :.2f}")

# Plot outputs
# 11. 결과값 산포도와 라인 겹쳐서 그래프 그리기
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

# plt.show()
plt.savefig(r"res/lr_image.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("res/lr_image.png")

res.close()
