# 0. 패키지 로드
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import time

# res = open("res/sk_3_result.txt", 'w')

# res.write(f"sk_3.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}\n\n")
print(f"sk_3.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}")

# res.write("K-최근접 이웃 실습\n\n")
print("K-최근접 이웃 실습")

# 0-1. 실행 결과가 일관되게 나오도록 하기 위해 시드 설정
np.random.seed(0)

# 1. 입력, 출력, 시간축 데이터 생성
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]  # 시간축 데이터
y = np.sin(X).ravel()

# Add noise to targets
# 2. 노이즈 주기
y[::5] += 1 * (0.5 - np.random.rand(8))

# 3. k 값 설정
n_neighbors = 5

for i, weights in enumerate(["uniform", "distance"]):  # 4. 가중치 설정
	# 5. 모델 로드
	knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)

	# 6. 모델 훈련
	# y_ = knn.fit(X, y).predict(T)  # 이런 방식보다 아래처럼 나누어 쓰는 것을 권장
	knn.fit(X, y)
	y_ = knn.predict(T)

	# 7. 입출력값/예측값 + 산포도 라인 출력
	plt.subplot(2, 1, i + 1)
	plt.scatter(X, y, color="darkorange", label="data")
	plt.plot(T, y_, color="navy", label="prediction")
	plt.axis("tight")
	plt.legend()
	plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
# plt.show()
plt.savefig(r"res/sk_3_image.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("res/sk_3_image.png")

# res.close()
