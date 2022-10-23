import pandas as pd
import numpy as np

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'], 'B': ['B0', 'B1', 'B2'], 'C': ['C0', 'C1', 'C2']}, index=[0, 1, 2])
df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'], 'B': ['B3', 'B4', 'B5'], 'C': ['C3', 'C4', 'C5']}, index=[3, 4, 5])
df3 = pd.DataFrame({'A': ['A6', 'A7', 'A8'], 'B': ['B6', 'B7', 'B8'], 'C': ['C6', 'C7', 'C8']}, index=[6, 7, 8])

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
print(con_r.loc['z'], end="\n\n")
print(con_r.loc['z'].loc[6], end="\n\n")

df4 = pd.DataFrame({'A': ['A1', 'A5', 'A6'], 'B': ['B1', 'B5', 'B6'], 'C': ['C1', 'C5', 'C6']}, index=[1, 5, 6])
