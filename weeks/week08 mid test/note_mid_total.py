# week 2 2주

# 넘파이 맛보기
from numpy import random

x = random.randn()

print(x)


# 클래스와 인스턴스
class MyPerson:
	i = 5

	def __init__(self, name, age):
		self.name = name
		self.age = age

	def asksomething(self):
		print("what is your name?")


cls = MyPerson("James", 20)
print(cls)
print(cls.name)
print(cls.age)
cls.asksomething()


# # 모듈
# from dummy1 import datamining as dm
# # from : 폴더명
# # import : 함수 가져오기
# # as : 별칭 지정

# print(dm.mul3(300))


# as가 왜 필요한가?
# 여러 개의 모듈을 쓸 때 이름이 겹칠 수 있음
from numpy import random  # 빠르지만 외부 라이브러리를 사용하기 때문에 추가 용량 필요
import random as pyrd  # 비교적 느리지만 내장 라이브러리 사용

print(random.randn())
print(pyrd.random())


# sys.path 예제
import sys

for p in sys.path:
	print(p)


# function
# 꼭 맨 위에 정의할 필요는 없으나 import와 함께 파일의 위쪽에 모아두는 것을 권장
# 파이썬 기본 내장 함수 : built-in function
def func1():
	print("this is a user defined function")


# nested function
print(sum([1, 2, 3, 4, 5]))
print(max(15, 6))


# 메소드와 함수
# 메소드 : 클래스 안에 포함됨
# 함수 : 클래스 밖에 정의됨
class MyMath:
	def add(self, a, b):
		return a + b


p1 = MyMath()
print(p1.add(3, 5))

##################################

# week 3 3주

# Numpy Array Object
import numpy as np

# x = [1, 2, 3, 4, 5]
x = list(range(1, 8))
# x_np = np.array(x)
x_np = np.arange(1, 8)
x_np_1 = np.ones(8)
x_np_0 = np.zeros(8)
print(f"np.ones(8)\n{x_np_1}\n")

print(f"list(range(1, 8))\n{x}\n")
print(f"np.arange(1, 8)\n{x_np}\n")

print(f"type(x)\n{type(x)}\n")
print(f"x_np.dtype\n{x_np.dtype}\n")

print(f"sum(x)\n{sum(x)}\n")
print(f"x_np.sum()\n{x_np.sum()}\n")

print(f"x[0:5]\n{x[0:5]}\n")
print(f"x_np[0:5].sum()\n{x_np[0:5].sum()}\n")

##

x2 = [[1, 2, 3], [4, 5, 6]]
x2_np = np.array(x2)
print(f"x2\n{x2}\n")
print(f"x2_np\n{x2_np}\n")

print(f"x_np.shape\n{x_np.shape}\n")
print(f"x2_np.shape\n{x2_np.shape}\n")
# print(x2.shape)  # error
print(f"x2.shape\n({len(x2)}, {len(x2[0])})\n")

##

x3_np = np.empty((6, 2))  # 빈 배열 생성, 아무 값이나 들어간다. 없는 것으로 취급해야 하며 이 상태로 출력했을 때 나오는 것은 메모리에 있던 잔해
x4_np = np.empty(shape=(6, 2), dtype=int)
print(f"x4_np = np.empty(shape=(6, 2))\n{x4_np}\n")

x_len = x3_np.shape
for i in range(x_len[0]):
	x3_np[i] = (1, -1)
print("x3_np")
print(x3_np, end="\n\n")

for i in range(x_len[0]):
	x3_np[i] = 3
print(x3_np, end="\n\n")

x5_np = np.eye(5, dtype=int)
print("np.eye(5, dtype=int)")
print(x5_np, end="\n\n")

x6_np = np.eye(4, k=1)  # k만큼 대각선이 밀림
print(x6_np, end="\n\n")

for i in range(-5, 6):
	x__np = np.eye(4, k=i)  # array의 범위를 초과해도 오류는 나지 않지만 대각선이 배열 내에서 사라질 수 있음
	print(x__np, end="\n\n")

x6_np = np.linspace(2, 3, num=5)
print(x6_np, end="\n\n")

x6_np = np.linspace(2, 3, num=5, endpoint=False)
print(x6_np, end="\n\n")

x7_np = np.arange(1, 7)
print(x7_np, end="\n\n")
print(x7_np.reshape((2, 3)), end="\n\n")  # 원래의 배열을 바꾸지 않음, 사이즈 안 맞으면 오류 남
print(x7_np, end="\n\n")

x7_np = x7_np.reshape((2, 3))
print(x7_np, end="\n\n")
x7_np = x7_np.reshape(-1)
print(x7_np, end="\n\n")

x7_np = np.arange(1, 25)
print(x7_np, end="\n\n")
x7_np = x7_np.reshape((2, 3, 4))
print(x7_np, end="\n\n")
x7_np = x7_np.reshape(-1)
print(x7_np, end="\n\n")
x7_np = x7_np.reshape(3, -1)
print(x7_np, end="\n\n")
print(x7_np.shape, end="\n\n")

##

x8_np = np.arange(1, 25).reshape((1, 6, 4))
print(x8_np, end="\n\n")
print(x8_np.ndim, end="\n\n")  # 차원 수
print(x8_np.shape, end="\n\n")  # 차원 모양
print(x8_np.size, end="\n\n")  # 요소 개수
print(x8_np.dtype, end="\n\n")  # 요소 자료형
print(x8_np.itemsize, end="\n\n")  # 요소의 바이트 수
print(x8_np.strides, end="\n\n")  # 배열을 순회할 때 각 차원에서 단계별로 수행할 바이트의 튜플입니다. Tuple of bytes to step in each dimension when traversing an array.

x = [('f1', np.int16)]
print(np.dtype(x), end="\n\n")
x = [('f1', np.int16), ('f2', np.int32)]
print(np.dtype(x), end="\n\n")
y = [('a', 'i2'), ('b', 'S2')]
print(np.dtype(y), end="\n\n")

print(np.dtype('i4, (2,3)f8'), end="\n\n")

print(np.dtype((np.int16, {'x': (np.int8, 0), 'y': (np.int8, 1)})), end="\n\n")
# print(np.dtype({'name': ['gender', 'age'], 'format': ['S1', np.uint8]}))
print(np.dtype({'surname': ('S25', 0), 'age': (np.uint8, 25)}), end="\n\n")

a = np.float32(5)
print(a, end="\n\n")
print(type(a), end="\n\n")
print(a.dtype, end="\n\n")

b = np.int_([4.0, 7.0, 999.9])
print(b, end="\n\n")
print(type(b), end="\n\n")
print(b.dtype, end="\n\n")

c = np.arange(7, dtype=np.uint16)
print(c, end="\n\n")
print(type(c), end="\n\n")
print(c.dtype, end="\n\n")

dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
print(dt['name'], end="\n\n")
print(dt['grades'], end="\n\n")

arr = np.array([('jin', 25, 67), ('suho', 18, 77)], dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
print(arr, end="\n\n")
print(arr[1], end="\n\n")
print(arr['age'], end="\n\n")
arr['age'][1] = 20
# arr['age'] = 20
arr['weight'][0] = 65
print(arr, end="\n\n")

print(np.dtype([("", np.int16), ("", np.float32)]), end="\n\n")

np.dtype({'names': ['col1', 'col2'], 'formats': ['i4', 'f4']})
np.dtype({'names': ['col1', 'col2'], 'formats': ['i4', 'f4'], 'offsets': [0, 4], 'itemsize': 12})

np.dtype({'col1': ('i1', 0), 'col2': ('f4', 1)})


def print_offsets(d):
	print("offsets:", [d.fields[name][1] for name in d.names])
	print("itemsize:", d.itemsize)


print_offsets(np.dtype('u1,u1,i4,u1,i8,u2'))

d = np.dtype('u1,u1,i4,u1,i8,u2')
print(d, end="\n\n")
print(d.itemsize, end="\n\n")
print(d.fields, end="\n\n")
print(d.names, end="\n\n")
print(d.fields['f0'], end="\n\n")
print(d.fields['f0'][1], end="\n\n")

print_offsets(np.dtype('u1,u1,i4,u1,i8,u2', align=True))

a = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')
a[1] = (7, 8, 9)
print(a, end="\n\n")

a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])
b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])
print(a, end="\n\n")
print(b.dtype, end="\n\n")
b[:] = a
print(b.dtype)  # 데이터타입은 안 변하고 값만 바꿈

arr = np.array([[1, 2], [3, 4], [5, 6]])
print(arr, end="\n\n")
print(arr[[0, 1, 2], [0, 1, 0]], end="\n\n")  # 인덱싱으로 인덱싱하기

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([11, 12, 13])
# print(arr1 + arr2)  # ValueError: operands could not be broadcast together with shapes (5,) (3,)
# Dimension이 같은 경우: 각 차원별로 크기가 같거나, 다르다면 어느 한 쪽이 1이어야 함
# Dimension이 다른 경우: 둘 중 차원이 작은 것의 크기가 𝑎1 × 𝑎2 × ⋯ × 𝑎𝑛일 때, 차원이 같아지도록 차이 나는 개수만큼 앞을 1로 채워 1 × ⋯ 1 × 𝑎1 × 𝑎2 × ⋯ × 𝑎𝑛와 같이 만든 후 Dimension이 같은 경우와 동일한 조건을 만족하여야 함
arr1_nx = arr1[:, np.newaxis]
arr2_nx = arr2[:, np.newaxis]
print(arr1_nx.shape, end="\n\n")
print(arr2_nx.shape, end="\n\n")
print(arr1_nx + arr2, end="\n\n")
arr2_1 = arr2_nx + arr1
arr2_1 = arr2_1.T  # transpose
print(arr2_1, end="\n\n")

################################

# week 4 4주

"""
week 4
"""
import numpy as np

print(np.add.nin, end="\n\n")  # 입력 개수
print(np.add.nout, end="\n\n")  # 출력 개수

print(np.exp.nin, end="\n\n")
print(np.exp.nout, end="\n\n")

print(np.ufunc.nin, end="\n\n")
print(np.ufunc.nout, end="\n\n")
# print(np.sum.nout)  # 모든 함수가 nin을 갖지는 않음

print(np.add([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), end="\n\n")

arr = np.array([1, 2, 3, 4, 5])
print(np.add.reduce(arr), end="\n\n")

arr = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
print(arr, end="\n\n")
print(arr.shape, end="\n\n")
print(np.add.reduce(arr, axis=0), end="\n\n")
print(np.add.reduce(arr, axis=1), end="\n\n")

print(np.add.accumulate([1, 2, 3, 4, 5]), end="\n\n")  # 누적 합
print(np.add.accumulate([1, 1, 1, 1, 1]), end="\n\n")  # 누적 합

arr = np.arange(1, 13).reshape(3, 4)
print(arr, end="\n\n")
print(arr.shape, end="\n\n")
print(np.add.accumulate(arr), end="\n\n")  # 위에서 아래로 누적 합
print(np.add.accumulate(arr, axis=0), end="\n\n")  # 위에서 아래로 누적 합
print(np.add.accumulate(arr, axis=1), end="\n\n")  # 위에서 아래로 누적 합

arr = np.arange(1, 13).reshape(3, 4)
print(arr, end="\n\n")
print(arr.shape, end="\n\n")
print(np.multiply.accumulate(arr), end="\n\n")  # 위에서 아래로 누적 합
print(np.multiply.accumulate(arr, axis=0), end="\n\n")  # 위에서 아래로 누적 합
print(np.multiply.accumulate(arr, axis=1), end="\n\n")  # 위에서 아래로 누적 합

arr = np.arange(0, 7)
print(arr, end="\n\n")
print(arr.shape, end="\n\n")
print(np.add.reduceat(arr, [0, 3, 5, 6]), end="\n\n")
print(np.add.reduceat(arr, [0, 1, 2, 3]), end="\n\n")
print(np.add.reduceat(arr, [1, 2, 3]), end="\n\n")
print(np.add.reduceat(arr, arr), end="\n\n")
# 뭔지 모르겠음

arr = np.linspace(0, 5, 6).reshape(2, 3)
print(arr, end="\n\n")

print(np.multiply.outer([1, 2, 3], [1, 2, 3]), end="\n\n")
print(np.multiply.outer([1, 2, 4], [2, 4, 8]), end="\n\n")
print(np.multiply.outer([1, 2, 4], [2, 4, 8, 16]), end="\n\n")

print(np.subtract(1, 2), end="\n\n")  # 위에서 한 거 다 알아서 해보면 됨

print(np.power(2, 3), end="\n\n")
arr = np.array([1, 2, 3, 4, 5])
print(np.power(arr, 3), end="\n\n")

print(np.sin(np.pi / 2), end="\n\n")
print(np.cos(np.pi / 2), end="\n\n")  # 사실상 0이라는 뜻
print(np.cos(np.deg2rad(90)), end="\n\n")

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

print(np.bitwise_and(7, 5), end="\n\n")  # 7 = 111, 5 = 0101
print(np.bitwise_and(8, 5), end="\n\n")  # 8 = 1000
# 7 & 5 = 0101
# 7 & 8 = 0000
# 9 = 1001
# 9 & 5 = 0001
print(np.bitwise_and(9, 5), end="\n\n")
print(np.bitwise_or(9, 5), end="\n\n")  # 1101 = 13
print(np.binary_repr(100), end="\n\n")
print(np.binary_repr(7567), end="\n\n")
print(np.binary_repr(75678), end="\n\n")

print(np.bitwise_and([7, 8, 9, 13, 45678], 5), end="\n\n")

arr = np.random.randn(3, 4)
print(arr, end="\n\n")
arr1 = arr[0, :]
print(arr1, end="\n\n")
arr2 = arr1
print(arr2, end="\n\n")
print('---')
arr1[:] = 0
print(arr, '\n', arr1, '\n', arr2, end="\n\n")

arr1 = arr[0, :].copy()
print(arr1, end="\n\n")
arr2 = arr1.copy()
print(arr2, end="\n\n")
print('---')
arr1[:] = 9
print(arr, '\n', arr1, '\n', arr2, end="\n\n")

a = np.array([1, 2, 3])
b = np.array([2, 2, 3])
b1 = 2
arr = a * b
arr1 = a * b1
print(arr, end="\n\n")
print(arr1, end="\n\n")

arr1 = np.arange(4)
arr3 = np.ones(5)
print(arr1.shape, end="\n\n")
print(arr3.shape, end="\n\n")
arr2 = arr1.reshape(4, 1)
arr4 = np.ones((3, 4))
print(arr2.shape, end="\n\n")
print(arr4.shape, end="\n\n")
arr12 = arr1 + arr2
print(arr12, end="\n\n")
# arr13 = arr1 + arr3
# print(arr13)
arr23 = arr2 + arr3
print(arr23, end="\n\n")
# arr24 = arr2 + arr4
# print(arr24)
arr14 = arr1 + arr4
print(arr14, end="\n\n")

arr1 = np.array([1, 2, 3])
arr2 = np.array([[4], [5], [6]])
arr = np.broadcast(arr1, arr2)
print(arr, end="\n\n")
print(arr.numiter, end="\n\n")
print(arr.index, end="\n\n")
print(arr1 + arr2, end="\n\n")

import numpy as np
import matplotlib.pyplot as plt

arr1 = np.arange(10)
arr2 = np.arange(7)
arr_img = np.sqrt(arr1[:, np.newaxis] ** 2 + arr2 ** 2)
plt.pcolor(arr_img)
plt.colorbar()
plt.axis('equal')
# plt.show()
print(arr_img.mean(axis=0), end="\n\n")
print(arr_img.mean(axis=1), end="\n\n")
# print(arr_img.mean(axis=2))  # 차원이 있으면 연산 가능한 듯 함. 아주 중요하다고 함

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])
print(np.concatenate((arr1, arr2), axis=0), end="\n\n")
print(np.concatenate((arr1, arr2.T), axis=1), end="\n\n")
print(np.concatenate((arr1, arr2), axis=None), end="\n\n")

print(np.hstack((arr1, arr2.T)), end="\n\n")
print(np.vstack((arr1, arr2)), end="\n\n")

a = np.array([np.random.randint(0, 12) for _ in range(12)]).reshape(3, 4)
print(a, end="\n\n")
a.sort()
print(a, end="\n\n")
a1 = np.sort(a, axis=None)
# a.sort(axis=None)
print(a1, end="\n\n")
print(a, end="\n\n")
a1 = np.sort(a, axis=1)
a.sort(axis=1)
print(a1, end="\n\n")
print(a, end="\n\n")
a1 = np.sort(a, axis=0)
a.sort(axis=0)
print(a1, end="\n\n")
print(a, end="\n\n")

dtypes = [('name', 'S10'), ('height', float), ('age', int)]
values = [('Jin', 175, 59), ('Suho', 185, 19), ('Naeun', 162, 28), ('Naeun2', 162, 38), ('Naeun3', 1162, 28)]
arr = np.array(values, dtype=dtypes)  # 구조화된 배열 생성
print(np.sort(arr, order='height'), end="\n\n")
print(np.sort(arr, order='age'), end="\n\n")
print(np.sort(arr, order='name'), end="\n\n")
print(np.sort(arr, order=['age', 'height']), end="\n\n")

import matplotlib.pyplot as plt
import numpy as np

arr = np.array([15, 16, 16, 17, 19, 20, 22, 35, 43, 45, 55, 59, 60, 75, 88])
plt.hist(arr, bins=[0, 20, 40, 60, 80, 100])
plt.title("numbers depending on ages")
# plt.show()

arr1 = arr2 = arr3 = np.arange(0, 5, 1)
arr4 = np.array((arr1, arr2, arr3))
np.savetxt(r'weeks\week04\week04data\test1.txt', arr, delimiter=',')  # arr1은 배열
np.savetxt(r'weeks\week04\week04data\test2.txt', (arr1, arr2, arr3))  # 동일 크기의 2D 배열
np.savetxt(r'weeks\week04\week04data\test3.txt', arr1, fmt='%1.4e')  # 지수 표기
print(np.loadtxt(r'weeks\week04\week04data\test1.txt'), end="\n\n")
print(np.loadtxt(r'weeks\week04\week04data\test2.txt'), end="\n\n")
print(np.loadtxt(r'weeks\week04\week04data\test3.txt'), end="\n\n")
np.savetxt(r'weeks\week04\week04data\test1.csv', arr1, delimiter=',')  # arr1은 배열
np.savetxt(r'weeks\week04\week04data\test2.csv', arr4, delimiter=',')  # 동일 크기의 2D 배열
np.savetxt(r'weeks\week04\week04data\test3.csv', arr1, fmt='%1.4e')  # 지수 표기

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
print(a3, end="\n\n")

####################################

# week 5 5주

import pandas as pd
import numpy as np

data = np.random.random(4)
index = ['a', 'b', 'c', 'd']
ser = pd.Series(data, index=index)
data = [1, 1, 1, 2, 3, 3, 4, 2, 3, 4]
ser1 = pd.Series(data)

print(ser, end="\n\n")
print(ser1, end="\n\n")

print(ser1.values, end="\n\n")
print(ser1.index, end="\n\n")
print(ser1.value_counts(), end="\n\n\n")

dic = {"kimbap": "5000", "Sundae": "4000"}
ser2 = pd.Series(dic)
print(ser2, end="\n\n")
ser2 = pd.Series(dic, index=["a", "Sundae", "kimbap", "d"])
print(ser2, end="\n\n")

ser3 = pd.Series(42, index=["Answer", "of", "the", "Universe"])
print(ser3, end="\n\n")
print(ser2[0], end="\n\n")
print(ser2[1], end="\n\n")
print(ser2["a"], end="\n\n")
print(ser2["Sundae"], end="\n\n")

print(ser1[:4], end="\n\n")
print(ser1[::2], end="\n\n")
print(np.exp(ser1[::2]), end="\n\n")
print(np.log(ser1[::2]), end="\n\n")

ser4 = pd.Series(np.random.random(10))
ser5 = pd.Series(np.random.random(4))
ser4_1 = ser4[:4]
print(ser4_1 + ser5, end="\n\n")

ser5_1 = ser5[1:] + ser5[:-1]
print(ser5_1, end="\n\n")

ser6 = pd.Series(data, name="abcd")
print(ser6, end="\n\n")
ser6_1 = ser6.rename("dcba")
print(ser6_1.name, end="\n\n")

print("-----------------------------")

d = {"하나": [0, 1, 2, 4], "둘": [0, 1, 2, 3]}
df = pd.DataFrame(d)
print(df, end="\n\n")

df = pd.DataFrame(d, index=["b", "c", "a", "a"], columns=["영", "둘"])
print(df, end="\n\n")
print(df.index, end="\n\n")
print(df.columns, end="\n\n")

arr = np.zeros((2,), dtype=[('A', 'i4'), ('B', 'f4'), ('D', 'a10')])
arr[:] = [(1, 2, "Hello"), (2, 3, "World")]
df = pd.DataFrame(arr)
df1 = pd.DataFrame(arr, index=['first', 'second'])
df2 = pd.DataFrame(arr, columns=['C', 'A', 'B'])
print(df, end="\n\n")
print(df1, end="\n\n")
print(df2, end="\n\n")

data_dict = dict([('A', [1, 2, 3]), ('B', [4, 5, 6])])
df_o = pd.DataFrame(data_dict)

df = pd.DataFrame.from_dict(data_dict,
                            orient="index", columns=["one", "two", "three"])
print(data_dict, end="\n\n")
print(df_o, end="\n\n")
print(df, end="\n\n")

print(df["one"], end="\n\n")
print(df["two"], end="\n\n")
df["fourth"] = df["one"]
print(df, end="\n\n")

del df["one"]
df.pop("two")
print(df, end="\n\n")

df["random"] = "hello"
df["cut"] = df["three"][:1]
df.insert(2, "whatup", [100, 100])
print(df, end="\n\n")

ser = pd.Series([1, 2, 3], index=["a", "b", "c"])
print(ser.drop(labels=["b"]), end="\n\n")

print(df, end="\n\n")

print(df.loc["A"], end="\n\n")
print(df.iloc[0], end="\n\n")

df_1 = pd.DataFrame(np.arange(0, 20).reshape(5, 4), columns=["A", "B", "C", "D"])
df_2 = pd.DataFrame(np.ones(9).reshape(3, 3), columns=["A", "B", "C"])
print(df_1, end="\n\n")
print(df_2, end="\n\n")
print(df_1.add(df_2, fill_value=None), end="\n\n")
print(df_2 * 3 + 2, end="\n\n")

print(df_1, end="\n\n")
print(df_1.T, end="\n\n")

ser = pd.Series(np.random.randn(1000))
print(ser.head(), end="\n\n")
print(ser.tail(10), end="\n\n")

date_ind = pd.date_range('10/5/2022', periods=5)

df = pd.DataFrame(np.random.randn(5, 3), index=date_ind, columns=["A", "B", "C"])
print(df, end="\n\n")

df = pd.DataFrame(np.random.randn(3, 4), index=["a", "b", "c"], columns=["A", "B", "C", "D"])

print(df.loc["c"], end="\n\n")
print(df.iloc[2], end="\n\n")
row_c = df.iloc[2]
print(df["A"], end="\n\n")
col_A = df["A"]

print(df.sub(row_c, axis=1), end="\n\n")
print(df.sub(col_A, axis=0), end="\n\n")

df = pd.DataFrame(np.arange(0, 12).reshape(4, 3))
df1 = pd.DataFrame(np.zeros(16).reshape(4, 4))
df = df + df1
print(df, end="\n\n")
print(df.mean(0), end="\n\n")
print(df.mean(1), end="\n\n")

print(df.sum(0, skipna=False), end="\n\n")
print(df.sum(1, skipna=True), end="\n\n")

df = pd.DataFrame(np.arange(0, 12).reshape(4, 3))
print(df.std(0), end="\n\n")
print(df.std(1), end="\n\n")

print(df.cumsum(0), end="\n\n")
print(df.cumsum(1), end="\n\n")

ser = pd.Series(np.random.randn(500))
ser[20:500] = np.nan
ser[10:20] = 5
print(ser.nunique(), end="\n\n")

ser = pd.Series(np.random.randn(1000))
ser[::2] = np.nan
print(ser.idxmin(), end="\n\n")
print(ser.idxmax(), end="\n\n")
print(ser.describe(percentiles=[0.05, 0.10, 0.90]), end="\n\n")

ser = pd.Series(['a', 'a', 'c', 'c', 'b', np.nan, 'd', 'g'])
print(ser.describe(), end="\n\n")

df = pd.DataFrame({"x": ['a', 'a', 'c', 'c', 'b'], "y": np.random.random(5)})
print(df.describe(include="all"), end="\n\n")

print(df["y"].idxmin(), end="\n\n")
print(df["y"].idxmax(), end="\n\n")
df = pd.DataFrame(np.arange(0, 12).reshape(4, 3))
print(df.apply(np.mean), end="\n\n")

df = pd.DataFrame({"A": np.arange(-3, 2), "B": np.ones(5)})
print(df, end="\n\n")
df = df.transform({"A": np.abs, "B": lambda x: x + 1})
print(df, end="\n\n")

########################################

# week 6 6주

# 6주 main

import pandas as pd
import numpy as np
import os

if not os.path.exists("data/"):
	os.mkdir("data/")

df1 = pd.DataFrame(np.random.randn(5, 4), columns=list('ABCD'), index=pd.date_range('20190701', periods=5))
print(df1, end="\n\n")
print(df1.loc['20190702':'20190704'], end="\n\n")
print(df1.iloc[0], end="\n\n")
print(df1['A'], end="\n\n\n")


ser1 = pd.Series(np.random.randn(4), index=list('abcd'))
print(ser1, end="\n\n")
print(ser1.loc['b'], end="\n\n")
print(ser1.loc['c'], end="\n\n")

ser1.loc['c'] = 0
print(ser1, end="\n\n")

ser1[2:] = 30
print(ser1, end="\n\n")
print("----------------------------", end="\n\n")


df2 = pd.DataFrame(np.random.randn(5, 4), columns=list('ABCD'), index=list("abcde"))
print(df2, end="\n\n")
print(df2.loc[['a', 'b', 'd'], :], end="\n\n")
print(df2.loc[['a', 'b', 'd'], "B":"D"], end="\n\n")
print(df2.loc[['a', 'b', 'd'], ["B", "D"]], end="\n\n\n")

ser2 = pd.Series(list("abcde"), index=[0, 3, 2, 5, 4])
print(ser2, end="\n\n")
print(ser2.loc[0:2], end="\n\n")
print(ser2.iloc[0:2], end="\n\n\n")


ser3 = pd.Series(list("abcde"))
print(ser3, end="\n\n")
print(ser3.loc[0:2], end="\n\n")

ser4 = ser2.sort_index()
print(ser4, end="\n\n")

ser5 = ser2.sort_values()
print(ser5, end="\n\n")
print("-------------------", end="\n\n")


df3 = pd.DataFrame(np.random.randn(5, 4), columns=list(range(0, 8, 2)), index=list(range(0, 10, 2)))
print(df3, end="\n\n")
print(df3.loc[0:4, 2:6], end="\n\n")
print(df3.iloc[0:3, 1:4], end="\n\n")
print(df3.iloc[0:3], end="\n\n")
print(df3.iloc[:, 1:], end="\n\n")
print(df3.iloc[[0, 3], [1, 2]], end="\n\n\n")


df4 = pd.DataFrame(np.random.randn(5, 4), columns=list("ABCE"), index=list("abcde"))
print(df4, end="\n\n")
print(df4.loc[lambda df: df.A > 0], end="\n\n")
print(df4.loc[:, lambda df: df.loc["a"] > 0], end="\n\n")
print("------------------------------", end="\n\n")


ser6 = pd.Series(np.arange(3))
print(ser6, end="\n\n")
ser6[5] = 7  # 없는 인덱스에 값 넣으면 그 인덱스만 생김, append와 다름
print(ser6, end="\n\n")
for i in range(3, 5):
	ser6[i] = 0
print(ser6, end="\n\n\n")


df5 = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list("ABC"))
print(df5, end="\n\n")

df5.loc[:, "T"] = df5.loc[:, "A"]
print(df5, end="\n\n")

data1 = list("AWESOME")
df5.loc[:, "G"] = pd.Series(data1)
print(df5, end="\n\n")

data2 = [33, 33, 33, "S", 33]
df5.loc[0, :] = pd.Series(data2)
print(df5, end="\n\n")

df5.loc[0, :] = data2
print(df5, end="\n\n")
print(df5["A"], end="\n\n")
print(df5.A, end="\n\n")  # 영어만 가능
print(df5.isna(), end="\n\n")
print(df5.notna(), end="\n\n")
print("-----------------------", end="\n\n\n")


d1 = {"one": [1, 2, 3], "two": [4, 5, 6]}
df6 = pd.DataFrame(d1, index=list("abc"))
df7 = df6.copy()

df7.loc["d"] = np.nan
df7.iloc[1:2, 1:2] = np.nan
print(df6, end="\n\n")
print(df7, end="\n\n")

df67 = df6 + df7
df67["three"] = np.nan
print(df67, end="\n\n")
print(df67.sum(), end="\n\n")
print(df67.prod(), end="\n\n")
print(df67.mean(), end="\n\n")
print(df67.std(), end="\n\n")
print(df67.max(), end="\n\n")
print(df67["one"].sum(), end="\n\n")
print(df67["one"].quantile(), end="\n\n")
print(df67["one"].var(), end="\n\n\n")


arr = np.array(np.arange(0, 9).reshape(3, 3))
print(arr, end="\n\n")
print(arr[:, 0].sum(), end="\n\n")
print(arr[:, 0].mean(), end="\n\n")
print(arr[:, 0].std(), end="\n\n")
print(arr[:, 0].max(), end="\n\n\n")


print(df67.fillna(df67.mean()), end="\n\n")
print(df67.fillna(0), end="\n\n\n")


df8 = pd.DataFrame([[np.nan, 2, 0, np.nan], [3, 4, np.nan, 1], [np.nan, 5, np.nan, 2], [np.nan, 1, 2, 3]],
		   columns=list("ABCD"))
print(df8, end="\n\n")
print(df8.fillna(0), end="\n\n")
print(df8.fillna(method="ffill"), end="\n\n")  # 앞에서 가져올 값이 없으면 안 채워짐
print(df8.fillna(method="bfill"), end="\n\n")  # 뒤에서 가져올 값이 없으면 안 채워짐

val = {"A": 11, "C": 33, "D": 44}
print(df8.fillna(val, limit=1), end="\n\n")
print("----------------------", end="\n\n")


data = {'name': ['haena', 'naeun', 'una', 'bum', 'suho'],
	'age': [30, 27, 28, 23, 18],
	'address': ['dogok', 'suwon', 'mapo', 'ilsan', 'yeoyi'],
	'grade': ['A', 'B', 'C', 'B', 'A'],
	'score': [100, 88, 73, 83, 95]}

df9 = pd.DataFrame(data, columns=['name', 'age', 'address', 'score', 'grade'])
print(df9, end="\n\n")
print(df9.to_csv("data/student_grade.csv"), end="\n\n")

df99 = pd.read_csv("data/student_grade.csv", header=None, nrows=3)
print(df99, end="\n\n")

df99 = pd.read_csv("data/student_grade.csv", index_col=0)
print(df99, end="\n\n")
print(df99.iloc[0:5, 1:6], end="\n\n")

df99 = pd.read_csv("data/student_grade.csv", index_col=["age"])
print(df99, end="\n\n")

df999 = pd.read_csv("data/student_grade.csv", names=["No", "name", "age", "address", "score", "grade"])
print(df999, end="\n\n")
print(df999.iloc[1:6], end="\n\n")

df_sep = pd.read_csv("data/student_grade.csv", sep='|', index_col=0)
print(df_sep, end="\n\n")
print(df9.to_csv("data/student_grade_sep.csv", sep='|'), end="\n\n")

df_sep = pd.read_csv("data/student_grade_sep.csv", sep='|', index_col=0)
print(df_sep, end="\n\n\n")


dfj = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row1', 'row2'], columns=['col1', 'col2'])
print(dfj, end="\n\n")
print(dfj.to_json(), end="\n\n")
print(dfj.to_json(orient='split'), end="\n\n")
print(dfj.to_json(orient='columns'), end="\n\n")
print(dfj.to_json(orient='values'), end="\n\n")
print(dfj.to_json(orient='table'), end="\n\n")
print(dfj.to_json("data/happy_json.json"), end="\n\n")

dfjr = pd.read_json("data/happy_json.json")
print(dfjr, end="\n\n\n")


url = 'https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/'
dfh = pd.read_html(url)
print(dfh, end="\n\n\n")


dfhtml = pd.DataFrame(np.random.randn(5, 4))
print(dfhtml.to_html(), end="\n\n")
print("------------------------", end="\n\n")


df672 = pd.DataFrame(np.arange(0, 100).reshape(20, 5))
print(df672.quantile([0.25, 0.55, 0.75]), end="\n\n")  # 미리 만들어둔 배열에 넣고 싶다면 for로 일일이 옮기거나 그냥 결과로 반환된 배열만 쓰기

np22 = np.arange(0, 100).reshape(20, 5)
print(np.quantile(np22, [0.25, 0.55, 0.75]), end="\n\n")
print(np.quantile(np22[:, 0], [0.25, 0.55, 0.75]), end="\n\n")
print(np.var(np22), end="\n\n")

# 6주 사이킷런

from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

if not os.path.exists("data/"):
	os.mkdir("data/")

url = 'http://bit.ly/kaggletrain'
train = pd.read_csv(url)

feature_cols = ['Pclass', 'Parch']
X = train.loc[:, feature_cols]
y = train.Survived

print(X.head, end="\n\n")
print(y.head, end="\n\n")

logreg = LogisticRegression()
logreg.fit(X, y)
url_test = 'http://bit.ly/kaggletest'
test = pd.read_csv(url_test)
X_new = test.loc[:, feature_cols]
new_pred_class = logreg.predict(X_new)

print(new_pred_class[0:5], end="\n\n")

kaggle_data = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': new_pred_class}).set_index('PassengerId')
kaggle_data.to_csv('data/sub.csv')

train.to_pickle('data/train.pkl')
pd.read_pickle('data/train.pkl')

#################################

# week 7 7주

import pandas as pd
import numpy as np

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'C': ['C0', 'C1', 'C2']}, index=[0, 1, 2])
df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
                    'B': ['B3', 'B4', 'B5'],
                    'C': ['C3', 'C4', 'C5']}, index=[3, 4, 5])
df3 = pd.DataFrame({'A': ['A6', 'A7', 'A8'],
                    'B': ['B6', 'B7', 'B8'],
                    'C': ['C6', 'C7', 'C8']}, index=[6, 7, 8])

print(df1, end="\n\n")
print(df3, end="\n\n")

fr = [df1, df2, df3]
print(fr, end="\n\n")

con_r = pd.concat(fr)
print(con_r, end="\n\n")

con_r = pd.concat(fr, keys=['x', 'y', 'z'])
print(con_r, end="\n\n")
print(con_r.loc['z'], end="\n\n")
print(con_r.loc['z'].loc[6], end="\n\n")

df4 = pd.DataFrame({'A': ['A3', 'A5', 'A6'],
                    'B': ['B3', 'B5', 'B6'],
                    'C': ['C3', 'C5', 'C6']}, index=[3, 5, 6])
print(df4)

con_r1 = pd.concat([df1, df4], axis=1, sort=False)
print(con_r1, end="\n\n")
con_r1 = pd.concat([df1, df4], axis=1, sort=False, join='inner')
print(con_r1, end="\n\n")

df4 = pd.DataFrame({'A': ['A1', 'A5', 'A6'],
                    'B': ['B1', 'B5', 'B6'],
                    'C': ['C1', 'C5', 'C6']}, index=[1, 5, 6])
print(df4, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=1, sort=False, join='outer')
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=0, sort=False, join='outer')
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=0, sort=True, join='outer')
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=0, sort=True, join='inner')
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=1, ignore_index=True)
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=1, ignore_index=False)
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], ignore_index=True)
print(con_r1, end="\n\n\n")

con_r2 = df1.append(df2)
print(con_r2, end="\n\n")

con_r2 = df1.append(df4, sort=False)
print(con_r2, end="\n\n")

print("---------------------------\n")

df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                    'value': np.random.randn(4)})
df2 = pd.DataFrame({'key': ['B', 'D', 'D', 'E'],
                    'value': np.random.randn(4)})
mg_r = pd.merge(df1, df2, on='key')
print(mg_r, end="\n\n")

mg_r = pd.merge(df1, df2)
print(mg_r, end="\n\n")

mg_r = pd.merge(df1, df2, on='key', how='left')
print(mg_r, end="\n\n")

mg_r = pd.merge(df1, df2, on='key', how='right')
print(mg_r, end="\n\n")

mg_r = pd.merge(df1, df2, on='key', how='outer')
print(mg_r, end="\n\n")

left = pd.DataFrame({'key1': ['Z0', 'Z0', 'Z1', 'Z2'],
                     'key2': ['ZO', 'Z1', 'Z0', 'Z1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['Z0', 'Z1', 'Z1', 'Z2'],
                      'key2': ['ZO', 'Z0', 'Z0', 'Z0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

result = pd.merge(left, right, on=['key1', 'key2'])
print(result, end="\n\n")

result = pd.merge(left, right, on=['key1', 'key2'], how='left')
print(result, end="\n\n")

result = pd.merge(left, right, on=['key1', 'key2'], how='right')
print(result, end="\n\n")

result = pd.merge(left, right, on=['key1', 'key2'], how='outer')
print(result, end="\n\n")

print("---------------------------\n")

import datetime

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6,
                   'B': ['x', 'y', 'w'] * 8,
                   'C': ['ha', 'ha', 'ha', 'hi', 'hi', 'hi'] * 4,
                   'D': np.arange(24),
                   'E': [datetime.datetime(2020, i, 1) for i in range(1, 13)]
                        + [datetime.datetime(2020, i, 15) for i in range(1, 13)]})

print(df, end="\n\n")

pivot_r = pd.pivot_table(df, values='D', index=['A', 'B'], columns='C')
print(pivot_r, end="\n\n")

pivot_r = pd.pivot_table(df, values='D', index=['B'], columns=['A', 'C'], aggfunc=np.sum)
print(pivot_r, end="\n\n")

str_df = pivot_r.to_string(na_rep='')
print(str_df, end="\n\n")

print("---------------------------\n")

import re

m = re.search('(?<=abc)def', 'abcdef')
print(m.group(), end="\n\n")
print(m, end="\n\n")
print(m.group(0), end="\n\n")
# print("m.group(1)")
# print(m.group(1), end="\n\n")  # IndexError: no such group

m = re.search('(?<=-)\w+', 'spam-egg')
print(m.group(0), end="\n\n")
# print(m.group(1), end="\n\n")  # IndexError: no such group
