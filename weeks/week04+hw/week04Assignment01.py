import numpy as np

a = np.zeros(3, dtype=[('a', 'f2'), ('b', 'i2'), ('c', 'S2')])

# write under 8 lines -------------------------
a = np.zeros(10, dtype=[('a', 'f2'), ('b', 'i2'), ('c', 'S5')])
for i in range(0, 10):
	a[i][0] = (i + 1) ** 2
	a[i][1] = (i + 1) ** 5
	a[i][2] = 'a' * (i + 1)
# -------------------------

print(a)