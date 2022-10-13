import pandas as pd
import numpy as np
import os

if not os.path.exists("data/"):
	os.mkdir("data/")

df1 = pd.DataFrame(np.random.randn(5, 4), columns=list('ABCD'), index=pd.date_range('20190701', periods=5))
print("df1")
print(df1, end="\n\n")
print(df1.loc['20190702':'20190704'], end="\n\n")
print(df1.iloc[0], end="\n\n")
print(df1['A'], end="\n\n\n")


ser1 = pd.Series(np.random.randn(4), index=list('abcd'))
print("ser1")
print(ser1, end="\n\n")
print(ser1.loc['b'], end="\n\n")
print(ser1.loc['c'], end="\n\n")

ser1.loc['c'] = 0
print(ser1, end="\n\n")

ser1[2:] = 30
print(ser1, end="\n\n")
print("----------------------------", end="\n\n")


df2 = pd.DataFrame(np.random.randn(5, 4), columns=list('ABCD'), index=list("abcde"))
print("df2")
print(df2, end="\n\n")
print(df2.loc[['a', 'b', 'd'], :], end="\n\n")
print(df2.loc[['a', 'b', 'd'], "B":"D"], end="\n\n")
print(df2.loc[['a', 'b', 'd'], ["B", "D"]], end="\n\n\n")

ser2 = pd.Series(list("abcde"), index=[0, 3, 2, 5, 4])
print("ser2")
print(ser2, end="\n\n")
print(ser2.loc[0:2], end="\n\n")
print(ser2.iloc[0:2], end="\n\n\n")


ser3 = pd.Series(list("abcde"))
print("ser3")
print(ser3, end="\n\n")
print(ser3.loc[0:2], end="\n\n")

ser4 = ser2.sort_index()
print("ser4")
print(ser4, end="\n\n")

ser5 = ser2.sort_values()
print("ser5")
print(ser5, end="\n\n")
print("-------------------", end="\n\n")


df3 = pd.DataFrame(np.random.randn(5, 4), columns=list(range(0, 8, 2)), index=list(range(0, 10, 2)))
print("df3")
print(df3, end="\n\n")
print(df3.loc[0:4, 2:6], end="\n\n")
print(df3.iloc[0:3, 1:4], end="\n\n")
print(df3.iloc[0:3], end="\n\n")
print(df3.iloc[:, 1:], end="\n\n")
print(df3.iloc[[0, 3], [1, 2]], end="\n\n\n")


df4 = pd.DataFrame(np.random.randn(5, 4), columns=list("ABCE"), index=list("abcde"))
print("df4")
print(df4, end="\n\n")
print(df4.loc[lambda df: df.A > 0], end="\n\n")
print(df4.loc[:, lambda df: df.loc["a"] > 0], end="\n\n")
print("------------------------------", end="\n\n")


ser6 = pd.Series(np.arange(3))
print("ser6")
print(ser6, end="\n\n")
ser6[5] = 7  # 없는 인덱스에 값 넣으면 그 인덱스만 생김, append와 다름
print(ser6, end="\n\n")
for i in range(3, 5):
	ser6[i] = 0
print(ser6, end="\n\n\n")


df5 = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list("ABC"))
print("df5")
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
print("df6, df7")
print(df6, end="\n\n")
print(df7, end="\n\n")

df67 = df6 + df7
df67["three"] = np.nan
print("df67")
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
print("arr")
print(arr, end="\n\n")
print(arr[:, 0].sum(), end="\n\n")
print(arr[:, 0].mean(), end="\n\n")
print(arr[:, 0].std(), end="\n\n")
print(arr[:, 0].max(), end="\n\n\n")


print(df67.fillna(df67.mean()), end="\n\n")
print(df67.fillna(0), end="\n\n\n")


df8 = pd.DataFrame([[np.nan, 2, 0, np.nan], [3, 4, np.nan, 1], [np.nan, 5, np.nan, 2], [np.nan, 1, 2, 3]],
		   columns=list("ABCD"))
print("df8")
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
print("df9")
print(df9, end="\n\n")
print(df9.to_csv("data/student_grade.csv"), end="\n\n")

df99 = pd.read_csv("data/student_grade.csv", header=None, nrows=3)
print("df99")
print(df99, end="\n\n")

df99 = pd.read_csv("data/student_grade.csv", index_col=0)
print(df99, end="\n\n")
print(df99.iloc[0:5, 1:6], end="\n\n")

df99 = pd.read_csv("data/student_grade.csv", index_col=["age"])
print(df99, end="\n\n")

df999 = pd.read_csv("data/student_grade.csv", names=["No", "name", "age", "address", "score", "grade"])
print("df999")
print(df999, end="\n\n")
print(df999.iloc[1:6], end="\n\n")

df_sep = pd.read_csv("data/student_grade.csv", sep='|', index_col=0)
print("df_sep")
print(df_sep, end="\n\n")
print(df9.to_csv("data/student_grade_sep.csv", sep='|'), end="\n\n")

df_sep = pd.read_csv("data/student_grade_sep.csv", sep='|', index_col=0)
print("df_sep")
print(df_sep, end="\n\n\n")


dfj = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row1', 'row2'], columns=['col1', 'col2'])
print("dfj")
print(dfj, end="\n\n")
print(dfj.to_json(), end="\n\n")
print(dfj.to_json(orient='split'), end="\n\n")
print(dfj.to_json(orient='columns'), end="\n\n")
print(dfj.to_json(orient='values'), end="\n\n")
print(dfj.to_json(orient='table'), end="\n\n")
print(dfj.to_json("data/happy_json.json"), end="\n\n")

dfjr = pd.read_json("data/happy_json.json")
print("dfjr")
print(dfjr, end="\n\n\n")


url = 'https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/'
dfh = pd.read_html(url)
print("dfh")
print(dfh, end="\n\n\n")


dfhtml = pd.DataFrame(np.random.randn(5, 4))
print(dfhtml.to_html(), end="\n\n")
print("------------------------", end="\n\n")


df672 = pd.DataFrame(np.arange(0, 100).reshape(20, 5))
print("df672")
print(df672.quantile([0.25, 0.55, 0.75]), end="\n\n")

np22 = np.arange(0, 100).reshape(20, 5)
print("np22")
print(np.quantile(np22, [0.25, 0.55, 0.75]), end="\n\n")
print(np.quantile(np22[:, 0], [0.25, 0.55, 0.75]), end="\n\n")
print(np.var(np22), end="\n\n")
