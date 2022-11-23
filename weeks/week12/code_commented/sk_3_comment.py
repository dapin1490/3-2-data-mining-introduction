# 0. 패키지 로드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# 1. 입력, 출력값, 시간축 데이터 생성
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# 2. 데이터내 노이즈 주입
y[::5] += 1 * (0.5 - np.random.rand(8))

# 3. K값 설정
n_neighbors = 3

# 4. 가중치 설정
# i=0
weights = "uniform"

# i=0
# weights="distance"


# 5. 모델 로드
knn = KNeighborsRegressor(n_neighbors, weights=weights)

# 6. 모델 훈련
knn.fit(X, y)

# 7. 모델 예측
y_ = knn.predict(T)

# 8. 입출력값/예측값 산포도+라인 겹쳐서 출력
plt.scatter(X, y, color="darkorange", label="data")
plt.plot(T, y_, color="navy", label="prediction")
plt.axis("tight")
plt.legend()
plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))
# plt.tight_layout()
plt.show()
