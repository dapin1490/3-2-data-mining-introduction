# 0. 패키지 로드
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 1. 데이터 로드
X, y = load_iris(return_X_y=True)

# 2. 훈련/테스트 데이터셋 구분
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Naive Bayes 모델 로드
gnb = GaussianNB()

# 4. 모델 훈련
gnb.fit(X_train, y_train)

# 5. 모델 예측
y_pred = gnb.predict(X_test)

# 6. 결과 출력
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
