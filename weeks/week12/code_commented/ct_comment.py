#0. 패키지 로드
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

#1. 데이터 로드
iris = load_iris()

#2. 트리 기법 파라미터 세팅
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
#for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
pairidx = 0
pair = [0,1]

#3. 데이터 준비
X = iris.data[:, pair]
y = iris.target

#4. 모델 로드
clf = DecisionTreeClassifier()

#5. 모델 훈련
clf.fit(X, y)


#6. 결과 바운더리 출력
#ax = plt.subplot(2, 3, pairidx + 1)
#plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.RdYlBu,
    response_method="predict",
#    ax=ax,
    xlabel=iris.feature_names[pair[0]],
    ylabel=iris.feature_names[pair[1]],
)

#7. 훈련 샘플 출력
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

# %%
# Display the structure of a single decision tree trained on all the features
# together.
from sklearn.tree import plot_tree

#8. 트리 구조 출력
plt.figure()
#clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.title("Decision tree trained on all the iris features")
plt.show()
