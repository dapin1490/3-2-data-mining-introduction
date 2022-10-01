import numpy as np

data_len = int(input("길이를 입력하시요.:"))
x = np.random.randn(data_len)

# write under 5 lines -------------------------
x = x.reshape((10, 10, data_len // 100))
np.savetxt('test.csv', x.mean(axis=2), delimiter=',')
# -------------------------