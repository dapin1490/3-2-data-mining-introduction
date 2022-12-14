"""
나일강 흐름 관련 데이터인 nile.csv가 존재한다. 주어진 데이터는 나일강의 수심 깊이를 측정한 것이다. 해당 데이터를 histogram화 시키고 이와 비슷하게 도출되는 정규분포 식을 겹치게 하여 시각적으로 비교되게끔 plot하시요. 다만 하기 조건의 유의하시요. (5점)

- 정규분포식을 직접 작성하시기 보다는 아래와 같은 코드를 활용하시요

import scipy.stats as stats
…………
df_lins = np.linspace(df_min,df_max, 100)
df_norm = stats.norm.pdf(df_lins, df_mean, df_std)

상기에서 df_min과 df_max의 경우는 데이터내 최저점, 최고점을 뜻하며, df_mean과 df_std는 데이터 평균 및 표준편차를 의미한다.

- plot이 두번 겹쳐 보이게 하기 위해서는 plt.plot()을 두번 연속으로 한 이후에 plt.show()를 하면 겹치게 보이는 점을 참고하세요.

(5점)
"""
# 풀이 실패

import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"weeks\week15\data\nile.csv")
df_min = data[' "Flood"'].min()
df_max = data[' "Flood"'].max()
df_mean = data[' "Flood"'].mean()
df_std = data[' "Flood"'].std()

df_lins = np.linspace(df_min, df_max, 100)
df_norm = stats.norm.pdf(df_lins, df_mean, df_std)

plt.hist(' "Flood"', data=data)
plt.hist(df_norm + df_mean)
plt.show()
