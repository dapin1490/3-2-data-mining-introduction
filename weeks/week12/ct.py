# 라이브러리
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import plot_tree

import time

# res = open("res/ct_result.txt", 'w')

print(f"ct.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}")
# res.write(f"ct.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}\n\n")

# res.write("분류 트리 실습\n")
# res.write("Pruning : 가지 치기\n")
# res.write("여기서는 의사 결정 트리를 쓰지만(나무 하나), 수많은 나무를 이용하는 랜덤 포레스트도 있다.\n")

# 데이터 로드
iris = load_iris()

# Parameters
# 파라미터 세팅
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

plt.figure(figsize=(12, 9))

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
	# We only take the two corresponding features
	# 데이터 준비
	X = iris.data[:, pair]
	y = iris.target

	# Train
	# 모델 로드
	clf = DecisionTreeClassifier()

	# 모델 훈련
	clf.fit(X, y)

	# Plot the decision boundary
	# 결과 바운더리 출력
	ax = plt.subplot(2, 3, pairidx + 1)
	plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
	DecisionBoundaryDisplay.from_estimator(
		clf,
		X,
		cmap=plt.cm.RdYlBu,
		response_method="predict",
		ax=ax,
		xlabel=iris.feature_names[pair[0]],
		ylabel=iris.feature_names[pair[1]],
	)

	# Plot the training points
	# 학습 샘플 출력
	for i, color in zip(range(n_classes), plot_colors):
		idx = np.where(y == i)
		plt.scatter(
			X[idx, 0],
			X[idx, 1],
			c=color,
			label=iris.target_names[i],
			cmap=plt.cm.RdYlBu,
			edgecolor="black",
			s=15,
		)

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")
# plt.show()
plt.savefig(r"res/ct_image01.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("res/ct_image01.png")

# 트리 구조 출력
plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.title("Decision tree trained on all the iris features")
# plt.show()
plt.savefig(r"res/ct_image02.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("res/ct_image02.png")

# res.close()
