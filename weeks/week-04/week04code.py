"""
week 4
"""
import numpy as np

print(np.add.nin)  # 입력 개수
print(np.add.nout)  # 출력 개수
print()
print(np.exp.nin)
print(np.exp.nout)
print()
print(np.ufunc.nin)
print(np.ufunc.nout)
print()
# print(np.sum.nout)  # 모든 함수가 nin을 갖지는 않음

print(np.add([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
print()

arr = np.array([1, 2, 3, 4, 5])
print(np.add.reduce(arr))
print()

arr = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
print(arr)
print(arr.shape)
print(np.add.reduce(arr, axis=0))
print(np.add.reduce(arr, axis=1))
print()

print(np.add.accumulate([1, 2, 3, 4, 5]))  # 누적 합
print(np.add.accumulate([1, 1, 1, 1, 1]))  # 누적 합
print()

arr = np.arange(1, 13).reshape(3, 4)
print(arr)
print(arr.shape)
print(np.add.accumulate(arr))  # 위에서 아래로 누적 합
print(np.add.accumulate(arr, axis=0))  # 위에서 아래로 누적 합
print(np.add.accumulate(arr, axis=1))  # 위에서 아래로 누적 합
print()

arr = np.arange(1, 13).reshape(3, 4)
print(arr)
print(arr.shape)
print(np.multiply.accumulate(arr))  # 위에서 아래로 누적 합
print(np.multiply.accumulate(arr, axis=0))  # 위에서 아래로 누적 합
print(np.multiply.accumulate(arr, axis=1))  # 위에서 아래로 누적 합
print()

arr = np.arange(0, 7)
print(arr)
print(arr.shape)
print(np.add.reduceat(arr, [0, 3, 5, 6]))
print(np.add.reduceat(arr, [0, 1, 2, 3]))
print(np.add.reduceat(arr, [1, 2, 3]))
print(np.add.reduceat(arr, arr))
print()  # 뭔지 모르겠음

arr = np.linspace(0, 5, 6).reshape(2, 3)
print(arr)
print()

print(np.multiply.outer([1, 2, 3], [1, 2, 3]))
print(np.multiply.outer([1, 2, 4], [2, 4, 8]))
print(np.multiply.outer([1, 2, 4], [2, 4, 8, 16]))
print()

print(np.subtract(1, 2))  # 위에서 한 거 다 알아서 해보면 됨
print()

print(np.power(2, 3))
arr = np.array([1, 2, 3, 4, 5])
print(np.power(arr, 3))
print()

print(np.sin(np.pi / 2))
print(np.cos(np.pi / 2))  # 사실상 0이라는 뜻
print(np.cos(np.deg2rad(90)))
print()

##

import numpy as np
import matplotlib.pyplot as plt

arr = np.linspace(-np.pi, np.pi, 201)
plt.plot(arr, np.sin(arr))
plt.xlabel("Angle (rad)")
plt.ylabel('sin(x)')
plt.axis('tight')
# plt.show()

arr = np.linspace(2 * -np.pi, 2 * np.pi, 20)  # 샘플 수가 적으면 투박해짐 -> 좋은 거 아님
plt.plot(arr, np.sin(arr))
plt.xlabel("Angle (rad)")
plt.ylabel('sin(x)')
plt.axis('tight')
# plt.show()

arr = np.linspace(-2 * 1, 2 * 1, 201)
plt.plot(arr, np.sin(5 * np.pi * arr))
plt.xlabel("second")
plt.ylabel('sin(x)')
plt.axis('tight')
# plt.show()

print(np.bitwise_and(7, 5))  # 7 = 111, 5 = 0101
print(np.bitwise_and(8, 5))  # 8 = 1000
# 7 & 5 = 0101
# 7 & 8 = 0000
# 9 = 1001
# 9 & 5 = 0001
print(np.bitwise_and(9, 5))
print(np.bitwise_or(9, 5))  # 1101 = 13
print(np.binary_repr(100))
print(np.binary_repr(7567))
print(np.binary_repr(75678))
print()

print(np.bitwise_and([7, 8, 9, 13, 45678], 5))
print()

arr = np.random.randn(3, 4)
print(arr)
arr1 = arr[0, :]
print(arr1)
arr2 = arr1
print(arr2)
print('---')
arr1[:] = 0
print(arr, '\n', arr1, '\n', arr2)

arr1 = arr[0, :].copy()
print(arr1)
arr2 = arr1.copy()
print(arr2)
print('---')
arr1[:] = 9
print(arr, '\n', arr1, '\n', arr2)
print()

a = np.array([1, 2, 3])
b = np.array([2, 2, 3])
b1 = 2
arr = a * b
arr1 = a * b1
print(arr)
print(arr1)
print()

arr1 = np.arange(4)
arr3 = np.ones(5)
print(arr1.shape)
print(arr3.shape)
arr2 = arr1.reshape(4, 1)
arr4 = np.ones((3, 4))
print(arr2.shape)
print(arr4.shape)
arr12 = arr1 + arr2
print(arr12)
# arr13 = arr1 + arr3
# print(arr13)
arr23 = arr2 + arr3
print(arr23)
# arr24 = arr2 + arr4
# print(arr24)
arr14 = arr1 + arr4
print(arr14)
print()

arr1 = np.array([1, 2, 3])
arr2 = np.array([[4], [5], [6]])
arr = np.broadcast(arr1, arr2)
print(arr)
print(arr.numiter)
print(arr.index)
print(arr1 + arr2)
print()

import numpy as np
import matplotlib.pyplot as plt

arr1 = np.arange(10)
arr2 = np.arange(7)
arr_img = np.sqrt(arr1[:, np.newaxis] ** 2 + arr2 ** 2)
plt.pcolor(arr_img)
plt.colorbar()
plt.axis('equal')
# plt.show()
print(arr_img.mean(axis=0))
print(arr_img.mean(axis=1))
# print(arr_img.mean(axis=2))  # 차원이 있으면 연산 가능한 듯 함. 아주 중요하다고 함
print()

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])
print(np.concatenate((arr1, arr2), axis=0))
print(np.concatenate((arr1, arr2.T), axis=1))
print(np.concatenate((arr1, arr2), axis=None))
print()

print(np.hstack((arr1, arr2.T)))
print(np.vstack((arr1, arr2)))
print()

a = np.array([np.random.randint(0, 12) for _ in range(12)]).reshape(3, 4)
print(a)
a.sort()
print(a)
a1 = np.sort(a, axis=None)
# a.sort(axis=None)
print(a1)
print(a)
a1 = np.sort(a, axis=1)
a.sort(axis=1)
print(a1)
print(a)
a1 = np.sort(a, axis=0)
a.sort(axis=0)
print(a1)
print(a)
print()

dtypes = [('name', 'S10'), ('height', float), ('age', int)]
values = [('Jin', 175, 59), ('Suho', 185, 19), ('Naeun', 162, 28), ('Naeun2', 162, 38), ('Naeun3', 1162, 28)]
arr = np.array(values, dtype=dtypes)  # 구조화된 배열 생성
print(np.sort(arr, order='height'))
print(np.sort(arr, order='age'))
print(np.sort(arr, order='name'))
print(np.sort(arr, order=['age', 'height']))
print()

import matplotlib.pyplot as plt
import numpy as np

arr = np.array([15, 16, 16, 17, 19, 20, 22, 35, 43, 45, 55, 59, 60, 75, 88])
plt.hist(arr, bins=[0, 20, 40, 60, 80, 100])
plt.title("numbers depending on ages")
# plt.show()

arr1 = arr2 = arr3 = np.arange(0, 5, 1)
arr4 = np.array((arr1, arr2, arr3))
np.savetxt('test1.txt', arr, delimiter=',')  # arr1은 배열
np.savetxt('test2.txt', (arr1, arr2, arr3))  # 동일 크기의 2D 배열
np.savetxt('test3.txt', arr1, fmt='%1.4e')  # 지수 표기
print(np.loadtxt('test1.txt'))
print(np.loadtxt('test2.txt'))
print(np.loadtxt('test3.txt'))
np.savetxt('test1.csv', arr1, delimiter=',')  # arr1은 배열
np.savetxt('test2.csv', arr4, delimiter=',')  # 동일 크기의 2D 배열
np.savetxt('test3.csv', arr1, fmt='%1.4e')  # 지수 표기

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 데이터 로드
X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300)  # 모델 로드
clf.fit(X_train, y_train)  # 모델 훈련
clf.predict_proba(X_test[:1])  # 데이터 샘플 테스트
clf.predict(X_test[:5, :])  # 데이터 테스트
clf.score(X_test, y_test)  # 모델 정확률 계산

##

# 과제 힌트
a1 = np.array([0, 1, 2, 3])
a2 = np.array([3, 2, 1, 0])
a3 = np.append(a1, a2)
print(a3)
print()
