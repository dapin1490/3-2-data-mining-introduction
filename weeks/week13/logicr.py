# 패키지 로드
from sklearn.datasets import load_iris  # 데이터 로드
from sklearn.linear_model import LogisticRegression  # 모델 로드
import time

res = open("weeks/week13/res/logicr_result.txt", 'w')

print(f"logicr.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}")
res.write(f"logicr.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}\n\n")

res.write("로지스틱 회귀 실습\n\n")

# 데이터 로드
X, y = load_iris(return_X_y=True)

# 로지스틱 회귀 모델 로드 + 훈련
clf = LogisticRegression(random_state=0).fit(X, y)

# 모델 예측
y_res = clf.predict(X[:2, :])
res.write(f"predict : {y_res}\n")

# 모델 예측 확률 계산
pred_score = clf.predict_proba(X[:2, :])
res.write(f"clf.predict_proba(X[:2, :]) : {pred_score}\n")

# 정확도 계산
acc = clf.score(X, y)
res.write(f"acc = {acc :.2f}\n")

res.close()
