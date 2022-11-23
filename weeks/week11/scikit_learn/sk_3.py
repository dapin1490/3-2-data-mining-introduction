import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

n_neighbors = 5

for i, weights in enumerate(["uniform", "distance"]):
	knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
	y_ = knn.fit(X, y).predict(T)

	plt.subplot(2, 1, i + 1)
	plt.scatter(X, y, color="darkorange", label="data")
	plt.plot(T, y_, color="navy", label="prediction")
	plt.axis("tight")
	plt.legend()
	plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
# plt.show()
plt.savefig(r"res/scikit03.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("res/scikit03.png")
