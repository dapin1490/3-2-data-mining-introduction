import pandas as pd
import numpy as np

df = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'], 'B': ['one', 'two', 'one', 'one', 'two'], 'Data1': np.random.randn(5), 'Data2': np.random.randn(5)})

print(df, end="\n\n")

df1 = df.groupby('A')
gr_dict = dict(list(df1))

print(df1, end="\n\n")
print(gr_dict, end="\n\n")
print(gr_dict['ho'], end="\n\n")
print(df1.get_group('ho'), end="\n\n")
print(df.groupby(['A', 'B']).get_group(('ha', 'one')), end="\n\n")

df2 = df.groupby(['A', 'B']).get_group(('ha', 'one'))
# print(df2.mean(), end="\n\n")  # FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated.

df3 = df['Data2'].groupby(df['A'])
# print(df3.mean(), end="\n\n")
df3 = df['Data2'].groupby([df['A'], df['B']])
print(df3.groups, end="\n\n")

df2 = pd.DataFrame({'A': ['ho', 'hi', 'ha'], 'B': ['two', 'one', 'two'], 'Data1': np.random.randn(3)})

# print(df2.groupby(['A']).sum(), end="\n\n")
# print(df2.groupby('A', sort=False).sum(), end="\n\n")
# print(df2.groupby('A', sort=False).median(), end="\n\n")

arr = [['ha', 'ha', 'hi', 'hi', 'ho', 'ho'], ['one', 'two', 'one', 'one', 'two', 'two']]
ind = pd.MultiIndex.from_arrays(arr, names=['1st', '2nd'])
print(ind, end="\n\n")
ser = pd.Series(np.random.randn(6), index=ind)
print(ser, end="\n\n")
print(ser.groupby(["1st", "2nd"]).mean(), end="\n\n")
print(ser.groupby(level=0).mean(), end="\n\n")
print(ser.groupby(level=1).mean(), end="\n\n")
# print(ser.groupby(level=2).mean(), end="\n\n")  # IndexError: Too many levels: Index has only 2 levels, not 3

for name, group in df.groupby('A'):
	print(name)
	print(group)
print("\n\n")

df = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'], 'B': ['one', 'two', 'one', 'one', 'two'], 'Data1': np.ones(5), 'Data2': np.ones(5)})

df1 = df.groupby('A')
print(df1.groups, end="\n\n")
# print(df1.agg(sum), end="\n\n")

df2 = df.groupby(['A', 'B'])
df2_1 = df.groupby(['A', 'B'], as_index=False)

print(df2.agg(sum), end="\n\n")
print(df2_1.agg(sum), end="\n\n")
print(df2_1.size(), end="\n\n")
print(df2_1.count(), end="\n\n")

print(df1.agg({'Data2': np.sum, 'Data1': lambda x: np.sum(x)}), end="\n\n")
print(df1.agg({'Data2': np.sum, 'Data1': lambda x: np.mean(x)}), end="\n\n")

df1 = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'], 'B': ['one', 'two', 'one', 'one', 'two'], 'Data1': np.ones(5), 'Data2': np.ones(5)}, index=[0, 1, 2, 3, 4])
df2 = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'], 'B': ['one', 'two', 'one', 'one', 'two'], 'Data1': np.ones(5), 'Data2': np.ones(5)}, index=[0, 1, 2, 3, 4])
df3 = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'], 'B': ['one', 'two', 'one', 'one', 'two'], 'Data1': np.ones(5), 'Data2': np.ones(5)}, index=[0, 1, 2, 3, 4])

df_c = [df1, df2, df3]
df4 = pd.concat(df_c)
print(df4, end="\n\n")
print(df4.reset_index(), end="\n\n")

df4_1 = df4['Data1'].apply(lambda x: x / 100)
print(df4_1, end="\n\n")

df4['Data1_2'] = df4['Data1'].apply(lambda x: x / 100)
print(df4, end="\n\n")

df4 = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'], 'B': ['one', 'two', 'one', 'one', 'two'], 'Data1': np.random.randn(5), 'Data2': np.random.randn(5)})
df4['Data1_c'] = df4['Data1'].apply(lambda x: x / 100)
df4['Data1_d'] = df4['Data1'] > df4['Data1_c']
print(df4, end="\n\n")
print(df4.sort_values(by='Data1'), end="\n\n")
print(df4.sort_values(by='Data1_c'), end="\n\n")
print(df4.sort_values(by='Data1_d'), end="\n\n")

print(df4.head(), end="\n\n")
print(df4.tail(), end="\n\n")

ser = pd.Series([1, 2, 3, 4, 5, 6])
print(ser, end="\n\n")
print(ser.pct_change(), end="\n\n")
print(ser.pct_change(periods=3), end="\n\n")
print(ser.pct_change(periods=2), end="\n\n")

df = pd.DataFrame({'2018': [0.12, 0.24], '2019': [0.14, 0.26], '2020': [0.10, 0.22]}, index=['CO2', 'H2O'])
print(df, end="\n\n")
print(df.pct_change(), end="\n\n")
print(df.pct_change(axis='columns'), end="\n\n")
print(df.T.pct_change(), end="\n\n")

ser1 = pd.Series(np.random.randn(100))
ser2 = pd.Series(np.random.randn(100))
print(ser1.cov(ser2), end="\n\n")

df = pd.DataFrame(np.random.randn(10000, 4), columns=['a', 'b', 'c', 'd'])
print(df.cov(), end="\n\n")

print(df['a'].corr(df['b']), end="\n\n")
print(df['a'].corr(df['b'], method='spearman'), end="\n\n")
print(df.corr(), end="\n\n")
print(df['a'].corr(df['b'], method='kendall'), end="\n\n")
print(df['b'].corr(df['a'], method='kendall'), end="\n\n")
print(df.corr(method='spearman'), end="\n\n")

df = pd.DataFrame(np.random.randn(500, 3), columns=['a', 'b', 'c'])
df.iloc[::2] = np.nan
print(df.head(6), end="\n\n")
print(df.corr(), end="\n\n")

ind = ['a', 'b', 'c', 'd']
col = ['one', 'two', 'three']
df1 = pd.DataFrame(np.random.randn(4, 3), index=ind, columns=col)
df2 = pd.DataFrame(np.random.randn(4, 3), index=ind, columns=col)
print(df1.corrwith(df2), end="\n\n")
print(df2.corrwith(df1, axis=1), end="\n\n")

df1 = pd.DataFrame(np.random.randn(4, 3), index=ind, columns=col)
df2 = pd.DataFrame(np.random.randn(3, 3), index=ind[:3], columns=col)
print(df1.corrwith(df2), end="\n\n")
print(df2.corrwith(df1, axis=1), end="\n\n")

df1 = pd.DataFrame(np.random.randn(4, 3), index=ind, columns=col)
df2 = pd.DataFrame(np.random.randn(4, 3), index=ind, columns=col)
print(df1.corrwith(df2.T), end="\n\n")
print(df2.corrwith(df1.T, axis=1), end="\n\n")
# 요점 : 사이즈 맞추는 게 중요하다

ser = pd.Series(np.random.randn(5), index=list('abcde'))
print(ser, end="\n\n")
print(ser.rank(), end="\n\n")  # 크기순 순위

ser['d'] = ser['b']
print(ser, end="\n\n")
print(ser.rank(), end="\n\n")

s = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2020', periods=1000))
print(s, end="\n\n")
print(s.cumsum(), end="\n\n")
s1 = s.rolling(window=60)
print(type(s1), end="\n\n")
print(s1.mean(), end="\n\n")

import matplotlib.pyplot as plt

ser.plot(style='k')
# plt.show()
s1.mean().plot(style='k-')
# plt.show()

df = pd.DataFrame(np.random.randn(500, 3), index=pd.date_range('1/1/2020', periods=500), columns=['A', 'B', 'C'])
dfc = df.cumsum()
print(dfc.head(), end="\n\n")

df1 = dfc[:20]
print(df1.rolling(window=5).corr(df1['B']), end="\n\n")
print(df1.rolling(window=10).corr(df1['B']), end="\n\n")
print(df1.rolling(window=3).corr(df1['B']), end="\n\n")
print(df1.rolling(window=3).corr(df1['A']), end="\n\n")
print(df1.rolling(window=3).corr(df1['C']), end="\n\n")
