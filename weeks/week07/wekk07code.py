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

print("df1")
print(df1, end="\n\n")
print("df3")
print(df3, end="\n\n")

fr = [df1, df2, df3]
print("fr")
print(fr, end="\n\n")

con_r = pd.concat(fr)
print("con_r")
print(con_r, end="\n\n")

con_r = pd.concat(fr, keys=['x', 'y', 'z'])
print(con_r, end="\n\n")
print("con_r.loc['z']")
print(con_r.loc['z'], end="\n\n")
print("con_r.loc['z'].loc[6]")
print(con_r.loc['z'].loc[6], end="\n\n")

df4 = pd.DataFrame({'A': ['A3', 'A5', 'A6'],
                    'B': ['B3', 'B5', 'B6'],
                    'C': ['C3', 'C5', 'C6']}, index=[3, 5, 6])
print("df4")
print(df4)

con_r1 = pd.concat([df1, df4], axis=1, sort=False)
print("con_r1")
print(con_r1, end="\n\n")
con_r1 = pd.concat([df1, df4], axis=1, sort=False, join='inner')
print(con_r1, end="\n\n")

df4 = pd.DataFrame({'A': ['A1', 'A5', 'A6'],
                    'B': ['B1', 'B5', 'B6'],
                    'C': ['C1', 'C5', 'C6']}, index=[1, 5, 6])
print("df4")
print(df4, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=1, sort=False, join='outer')
print("pd.concat([df1, df4], axis=1, sort=False, join='outer')")
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=0, sort=False, join='outer')
print("pd.concat([df1, df4], axis=0, sort=False, join='outer')")
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=0, sort=True, join='outer')
print("pd.concat([df1, df4], axis=0, sort=True, join='outer')")
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=0, sort=True, join='inner')
print("pd.concat([df1, df4], axis=0, sort=True, join='inner')")
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=1, ignore_index=True)
print("pd.concat([df1, df4], axis=1, ignore_index=True)")
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], axis=1, ignore_index=False)
print("pd.concat([df1, df4], axis=1, ignore_index=False)")
print(con_r1, end="\n\n")

con_r1 = pd.concat([df1, df4], ignore_index=True)
print("pd.concat([df1, df4], ignore_index=True)")
print(con_r1, end="\n\n\n")

con_r2 = df1.append(df2)
print("df1.append(df2)")
print(con_r2, end="\n\n")

con_r2 = df1.append(df4, sort=False)
print("df1.append(df4, sort=False)")
print(con_r2, end="\n\n")

print("---------------------------\n")

df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                    'value': np.random.randn(4)})
df2 = pd.DataFrame({'key': ['B', 'D', 'D', 'E'],
                    'value': np.random.randn(4)})
mg_r = pd.merge(df1, df2, on='key')
print("pd.merge(df1, df2, on='key')")
print(mg_r, end="\n\n")

mg_r = pd.merge(df1, df2)
print("pd.merge(df1, df2)")
print(mg_r, end="\n\n")

mg_r = pd.merge(df1, df2, on='key', how='left')
print("pd.merge(df1, df2, on='key', how='left')")
print(mg_r, end="\n\n")

mg_r = pd.merge(df1, df2, on='key', how='right')
print("pd.merge(df1, df2, on='key', how='right')")
print(mg_r, end="\n\n")

mg_r = pd.merge(df1, df2, on='key', how='outer')
print("pd.merge(df1, df2, on='key', how='outer')")
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
print("pd.merge(left, right, on=['key1', 'key2'])")
print(result, end="\n\n")

result = pd.merge(left, right, on=['key1', 'key2'], how='left')
print("pd.merge(left, right, on=['key1', 'key2'], how='left')")
print(result, end="\n\n")

result = pd.merge(left, right, on=['key1', 'key2'], how='right')
print("pd.merge(left, right, on=['key1', 'key2'], how='right')")
print(result, end="\n\n")

result = pd.merge(left, right, on=['key1', 'key2'], how='outer')
print("pd.merge(left, right, on=['key1', 'key2'], how='outer')")
print(result, end="\n\n")

print("---------------------------\n")

import datetime

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6,
                   'B': ['x', 'y', 'w'] * 8,
                   'C': ['ha', 'ha', 'ha', 'hi', 'hi', 'hi'] * 4,
                   'D': np.arange(24),
                   'E': [datetime.datetime(2020, i, 1) for i in range(1, 13)]
                        + [datetime.datetime(2020, i, 15) for i in range(1, 13)]})

print("df")
print(df, end="\n\n")

pivot_r = pd.pivot_table(df, values='D', index=['A', 'B'], columns='C')
print("pd.pivot_table(df, values='D', index=['A', 'B'], columns='C')")
print(pivot_r, end="\n\n")

pivot_r = pd.pivot_table(df, values='D', index=['B'], columns=['A', 'C'], aggfunc=np.sum)
print("pd.pivot_table(df, values='D', index=['B'], columns=['A', 'C'], aggfunc=np.sum)")
print(pivot_r, end="\n\n")

str_df = pivot_r.to_string(na_rep='')
print("pivot_r.to_string(na_rep='')")
print(str_df, end="\n\n")

print("---------------------------\n")

import re

m = re.search('(?<=abc)def', 'abcdef')
print("m = re.search('(?<=abc)def', 'abcdef')")
print("m.group()")
print(m.group(), end="\n\n")
print("m")
print(m, end="\n\n")
print("m.group(0)")
print(m.group(0), end="\n\n")
# print("m.group(1)")
# print(m.group(1), end="\n\n")  # IndexError: no such group

m = re.search('(?<=-)\w+', 'spam-egg')
print("m = re.search('(?<=-)\w+', 'spam-egg')")
print("m.group(0)")
print(m.group(0), end="\n\n")
# print(m.group(1), end="\n\n")  # IndexError: no such group
