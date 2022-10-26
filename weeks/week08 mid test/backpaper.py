import numpy as np

data = [(22, 10, 6, 57, "ulsan"), (21, 10, 7, 56, "jeonbuk"), (10, 16, 12, 52, "daegu"), (11, 13, 14, 43, "seoul"), (11, 11, 16, 43, "suwon")]

# <------------여기서부터 수정 가능------------------>

a = np.zeros(5, dtype=[('win', 'i4'), ('draw', 'i4'), ('lose', 'i4'), ('score', 'i4'), ('team', 'S10')])
for i in range(0, len(data)):
	a[i] = data[i]

b = np.zeros(5, dtype=[('point', 'i4'), ('rate', 'f4'), ('team', 'S10')])

for i in range(0, len(data)):
	b[i][0] = a[i][0] * 3 + a[i][1] * 1
	b[i][1] = a[i][0] / (a[i][0] + a[i][1] + a[i][2])
	b[i][2] = a[i][4]

# <------------수정 가능 영역 끝------------------>

print(a)
print(b)
print(a.dtype)
print(b.dtype)


answer = """[(22, 10,  6, 57, b'ulsan') (21, 10,  7, 56, b'jeonbuk')
 (10, 16, 12, 52, b'daegu') (11, 13, 14, 43, b'seoul')
 (11, 11, 16, 43, b'suwon')]
[(76, 0.57894737, b'ulsan') (73, 0.55263156, b'jeonbuk')
 (46, 0.2631579 , b'daegu') (46, 0.28947368, b'seoul')
 (44, 0.28947368, b'suwon')]
[('win', '<i4'), ('draw', '<i4'), ('lose', '<i4'), ('score', '<i4'), ('team', 'S10')]
[('point', '<i4'), ('rate', '<f4'), ('team', 'S10')]"""

result = """[(22, 10,  6, 57, b'ulsan') (21, 10,  7, 56, b'jeonbuk')
 (10, 16, 12, 52, b'daegu') (11, 13, 14, 43, b'seoul')
 (11, 11, 16, 43, b'suwon')]
[(76, 0.57894737, b'ulsan') (73, 0.55263156, b'jeonbuk')
 (46, 0.2631579 , b'daegu') (46, 0.28947368, b'seoul')
 (44, 0.28947368, b'suwon')]
[('win', '<i4'), ('draw', '<i4'), ('lose', '<i4'), ('score', '<i4'), ('team', 'S10')]
[('point', '<i4'), ('rate', '<f4'), ('team', 'S10')]"""

print(len(answer) == len(result))
print(answer == result)
