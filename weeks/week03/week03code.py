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
print("np.eye(4, k=1)  # k만큼 대각선이 밀림")
print(x6_np, end="\n\n")

for i in range(-5, 6):
	x__np = np.eye(4, k=i)  # array의 범위를 초과해도 오류는 나지 않지만 대각선이 배열 내에서 사라질 수 있음
	print(f"np.eye(4, k={i})")
	print(x__np, end="\n\n")

x6_np = np.linspace(2, 3, num=5)
print("np.linspace(2, 3, num=5)")
print(x6_np, end="\n\n")

x6_np = np.linspace(2, 3, num=5, endpoint=False)
print("np.linspace(2, 3, num=5, endpoint=False)")
print(x6_np, end="\n\n")

x7_np = np.arange(1, 7)
print("np.arange(1, 7)")
print(x7_np, end="\n\n")
print("x7_np.reshape((2, 3))  # 원래의 배열을 바꾸지 않음, 사이즈 안 맞으면 오류 남")
print(x7_np.reshape((2, 3)), end="\n\n")  # 원래의 배열을 바꾸지 않음, 사이즈 안 맞으면 오류 남
print("x7_np")
print(x7_np, end="\n\n")

x7_np = x7_np.reshape((2, 3))
print("x7_np.reshape((2, 3))")
print(x7_np, end="\n\n")
x7_np = x7_np.reshape(-1)
print("x7_np.reshape(-1)")
print(x7_np, end="\n\n")

x7_np = np.arange(1, 25)
print("np.arange(1, 25)")
print(x7_np, end="\n\n")
x7_np = x7_np.reshape((2, 3, 4))
print("x7_np = x7_np.reshape((2, 3, 4))")
print(x7_np, end="\n\n")
x7_np = x7_np.reshape(-1)
print("x7_np = x7_np.reshape(-1)")
print(x7_np, end="\n\n")
x7_np = x7_np.reshape(3, -1)
print("x7_np = x7_np.reshape(3, -1)")
print(x7_np, end="\n\n")
print("x7_np.shape")
print(x7_np.shape, end="\n\n")

##

x8_np = np.arange(1, 25).reshape((1, 6, 4))
print("np.arange(1, 25).reshape((1, 6, 4))")
print(x8_np, end="\n\n")
print("print(x8_np.ndim)  # 차원 수")
print(x8_np.ndim, end="\n\n")  # 차원 수
print("print(x8_np.shape)  # 차원 모양")
print(x8_np.shape, end="\n\n")  # 차원 모양
print("print(x8_np.size)  # 요소 개수")
print(x8_np.size, end="\n\n")  # 요소 개수
print("print(x8_np.dtype)  # 요소 자료형")
print(x8_np.dtype, end="\n\n")  # 요소 자료형
print("print(x8_np.itemsize)  # 요소의 바이트 수")
print(x8_np.itemsize, end="\n\n")  # 요소의 바이트 수
print("print(x8_np.strides)  # 배열을 순회할 때 각 차원에서 단계별로 수행할 바이트의 튜플입니다. Tuple of bytes to step in each dimension when traversing an array.")
print(x8_np.strides, end="\n\n")  # 배열을 순회할 때 각 차원에서 단계별로 수행할 바이트의 튜플입니다. Tuple of bytes to step in each dimension when traversing an array.

x = [('f1', np.int16)]
print("x = [('f1', np.int16)]\nprint(np.dtype(x))")
print(np.dtype(x), end="\n\n")
x = [('f1', np.int16), ('f2', np.int32)]
print("x = [('f1', np.int16), ('f2', np.int32)]\nprint(np.dtype(x))")
print(np.dtype(x), end="\n\n")
y = [('a', 'i2'), ('b', 'S2')]
print("y = [('a', 'i2'), ('b', 'S2')]\nprint(np.dtype(y))")
print(np.dtype(y), end="\n\n")

print("print(np.dtype('i4, (2,3)f8'))")
print(np.dtype('i4, (2,3)f8'), end="\n\n")

print("print(np.dtype((np.int16, {'x': (np.int8, 0), 'y': (np.int8, 1)})))")
print(np.dtype((np.int16, {'x': (np.int8, 0), 'y': (np.int8, 1)})), end="\n\n")
# print(np.dtype({'name': ['gender', 'age'], 'format': ['S1', np.uint8]}))
print("print(np.dtype({'surname': ('S25', 0), 'age': (np.uint8, 25)}))")
print(np.dtype({'surname': ('S25', 0), 'age': (np.uint8, 25)}), end="\n\n")

a = np.float32(5)
print("a = np.float32(5)")
print(a, end="\n\n")
print("print(type(a))")
print(type(a), end="\n\n")
print("print(a.dtype)")
print(a.dtype, end="\n\n")

b = np.int_([4.0, 7.0, 999.9])
print("b = np.int_([4.0, 7.0, 999.9])")
print(b, end="\n\n")
print("print(type(b))")
print(type(b), end="\n\n")
print("print(b.dtype)")
print(b.dtype, end="\n\n")

c = np.arange(7, dtype=np.uint16)
print("c = np.arange(7, dtype=np.uint16)")
print(c, end="\n\n")
print("print(type(c))")
print(type(c), end="\n\n")
print("print(c.dtype)")
print(c.dtype, end="\n\n")

dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
print("dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])\nprint(dt['name'])")
print(dt['name'], end="\n\n")
print("print(dt['grades'])")
print(dt['grades'], end="\n\n")

arr = np.array([('jin', 25, 67), ('suho', 18, 77)], dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
print("arr = np.array([('jin', 25, 67), ('suho', 18, 77)], dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])")
print(arr, end="\n\n")
print("print(arr[1])")
print(arr[1], end="\n\n")
print("print(arr['age'])")
print(arr['age'], end="\n\n")
arr['age'][1] = 20
# arr['age'] = 20
arr['weight'][0] = 65
print("arr['age'][1] = 20\narr['weight'][0] = 65")
print("print(arr)")
print(arr, end="\n\n")

print("print(np.dtype([("", np.int16), ("", np.float32)]))")
print(np.dtype([("", np.int16), ("", np.float32)]), end="\n\n")

np.dtype({'names': ['col1', 'col2'], 'formats': ['i4', 'f4']})
np.dtype({'names': ['col1', 'col2'], 'formats': ['i4', 'f4'], 'offsets': [0, 4], 'itemsize': 12})

np.dtype({'col1': ('i1', 0), 'col2': ('f4', 1)})


def print_offsets(d):
	print("offsets:", [d.fields[name][1] for name in d.names])
	print("itemsize:", d.itemsize)


print_offsets(np.dtype('u1,u1,i4,u1,i8,u2'))

d = np.dtype('u1,u1,i4,u1,i8,u2')
print("d = np.dtype('u1,u1,i4,u1,i8,u2')")
print(d, end="\n\n")
print(d.itemsize, end="\n\n")
print(d.fields, end="\n\n")
print(d.names, end="\n\n")
print(d.fields['f0'], end="\n\n")
print(d.fields['f0'][1], end="\n\n")

print_offsets(np.dtype('u1,u1,i4,u1,i8,u2', align=True))

a = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')
a[1] = (7, 8, 9)
print("a = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')\na[1] = (7, 8, 9)")
print(a, end="\n\n")

a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])
b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])
print("a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])")
print(a, end="\n\n")
print("b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])\nprint(b.dtype)")
print(b.dtype, end="\n\n")
b[:] = a
print("b[:] = a\nprint(b.dtype)")
print(b.dtype)  # 데이터타입은 안 변하고 값만 바꿈

arr = np.array([[1, 2], [3, 4], [5, 6]])
print("arr = np.array([[1, 2], [3, 4], [5, 6]])")
print(arr, end="\n\n")
print("print(arr[[0, 1, 2], [0, 1, 0]])  # 인덱싱으로 인덱싱하기")
print(arr[[0, 1, 2], [0, 1, 0]], end="\n\n")  # 인덱싱으로 인덱싱하기

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([11, 12, 13])
print("arr1 = np.array([1, 2, 3, 4, 5])\narr2 = np.array([11, 12, 13])")
# print(arr1 + arr2)  # ValueError: operands could not be broadcast together with shapes (5,) (3,)
# Dimension이 같은 경우: 각 차원별로 크기가 같거나, 다르다면 어느 한 쪽이 1이어야 함
# Dimension이 다른 경우: 둘 중 차원이 작은 것의 크기가 𝑎1 × 𝑎2 × ⋯ × 𝑎𝑛일 때, 차원이 같아지도록 차이 나는 개수만큼 앞을 1로 채워 1 × ⋯ 1 × 𝑎1 × 𝑎2 × ⋯ × 𝑎𝑛와 같이 만든 후 Dimension이 같은 경우와 동일한 조건을 만족하여야 함
arr1_nx = arr1[:, np.newaxis]
arr2_nx = arr2[:, np.newaxis]
print("arr1_nx = arr1[:, np.newaxis]\narr2_nx = arr2[:, np.newaxis]")
print(arr1_nx.shape, end="\n\n")
print(arr2_nx.shape, end="\n\n")
print("print(arr1_nx + arr2)")
print(arr1_nx + arr2, end="\n\n")
arr2_1 = arr2_nx + arr1
arr2_1 = arr2_1.T  # transpose
print("arr2_1 = arr2_nx + arr1\narr2_1 = arr2_1.T  # transpose")
print(arr2_1, end="\n\n")
