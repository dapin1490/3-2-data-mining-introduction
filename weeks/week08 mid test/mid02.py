"""
2. 하기와 같은 코드가 존재한다. 해당 코드와 동일한 기능을 제공하는 코드를 수정 업데이트하여 제출하시요. 다만 아래의 조건을 따르시요. 참고로 코드 최적화를 위해서 list comprehension을 검색 및 활용 권장한다. (7점)
- 수정 업데이트시 수정 가능 영역 부분을 5줄 이하로 작성하여야 한다
- ;을 활용한 다중 코드 한줄화는 금지한다.

import numpy as np

data = [(22, 10, 6, 57, "ulsan"), (21, 10, 7, 56, "jeonbuk"), (10, 16, 12, 52, "daegu"), (11, 13, 14, 43, "seoul"), (11, 11, 16, 43, "suwon")]

<------------여기서부터 수정 가능------------------>

a = np.zeros(5, dtype=[('win', 'i4'), ('draw', 'i4'), ('lose', 'i4'), ('score', 'i4'), ('team', 'S10')])
for i in range(0, len(data)):
	a[i] = data[i]

b = np.zeros(5, dtype=[('point', 'i4'), ('rate', 'f4'), ('team', 'S10')])

for i in range(0, len(data)):
	b[i][0] = a[i][0] * 3 + a[i][1] * 1
	b[i][1] = a[i][0] / (a[i][0] + a[i][1] + a[i][2])
	b[i][2] = a[i][4]

<------------수정 가능 영역 끝------------------>

print(a)
print(b)
print(a.dtype)
print(b.dtype)
"""

import numpy as np

data = [(22, 10, 6, 57, "ulsan"), (21, 10, 7, 56, "jeonbuk"), (10, 16, 12, 52, "daegu"), (11, 13, 14, 43, "seoul"), (11, 11, 16, 43, "suwon")]

# < ------------여기서부터 수정 가능------------------>

a = np.array(data, dtype=[('win', 'i4'), ('draw', 'i4'), ('lose', 'i4'), ('score', 'i4'), ('team', 'S10')])
b = np.array([(a[i][0] * 3 + a[i][1] * 1, a[i][0] / (a[i][0] + a[i][1] + a[i][2]), a[i][4]) for i in range(len(data))], dtype=[('point', 'i4'), ('rate', 'f4'), ('team', 'S10')])

# < ------------수정 가능 영역 끝------------------>

print(a)
print(b)
print(a.dtype)
print(b.dtype)
