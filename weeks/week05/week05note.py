import pandas as pd
import numpy as np

data = np.random.random(4)
index = ['a', 'b', 'c', 'd']
ser = pd.Series(data, index=index)
data = [1, 1, 1, 2, 3, 3, 4, 2, 3, 4]
ser1 = pd.Series(data)

print("ser")
print(ser, end="\n\n")
print("ser1")
print(ser1, end="\n\n")

print(ser1.values, end="\n\n")
print(ser1.index, end="\n\n")
print(ser1.value_counts(), end="\n\n\n")

dic = {"kimbap": "5000", "Sundae": "4000"}
ser2 = pd.Series(dic)
print("ser2")
print(ser2, end="\n\n")
ser2 = pd.Series(dic, index=["a", "Sundae", "kimbap", "d"])
print(ser2, end="\n\n")

ser3 = pd.Series(42, index=["Answer", "of", "the", "Universe"])
print("ser3")
print(ser3, end="\n\n")
print("ser2")
print(ser2[0], end="\n\n")
print(ser2[1], end="\n\n")
print(ser2["a"], end="\n\n")
print(ser2["Sundae"], end="\n\n")

print("ser1")
print(ser1[:4], end="\n\n")
print(ser1[::2], end="\n\n")
print(np.exp(ser1[::2]), end="\n\n")
print(np.log(ser1[::2]), end="\n\n")

ser4 = pd.Series(np.random.random(10))
ser5 = pd.Series(np.random.random(4))
ser4_1 = ser4[:4]
print("ser4_1 + ser5")
print(ser4_1 + ser5, end="\n\n")

ser5_1 = ser5[1:] + ser5[:-1]
print("ser5_1")
print(ser5_1, end="\n\n")

ser6 = pd.Series(data, name="abcd")
print("ser6")
print(ser6, end="\n\n")
ser6_1 = ser6.rename("dcba")
print(ser6_1.name, end="\n\n")

print("-----------------------------")

d = {"하나": [0, 1, 2, 4], "둘": [0, 1, 2, 3]}
df = pd.DataFrame(d)
print("df")
print(df, end="\n\n")

df = pd.DataFrame(d, index=["b", "c", "a", "a"], columns=["영", "둘"])
print(df, end="\n\n")
print(df.index, end="\n\n")
print(df.columns, end="\n\n")

arr = np.zeros((2,), dtype=[('A', 'i4'), ('B', 'f4'), ('D', 'a10')])
arr[:] = [(1, 2, "Hello"), (2, 3, "World")]
df = pd.DataFrame(arr)
df1 = pd.DataFrame(arr, index=['first', 'second'])
df2 = pd.DataFrame(arr, columns=['C', 'A', 'B'])
print(df, end="\n\n")
print("df1")
print(df1, end="\n\n")
print("df2")
print(df2, end="\n\n")

data_dict = dict([('A', [1, 2, 3]), ('B', [4, 5, 6])])
df_o = pd.DataFrame(data_dict)

df = pd.DataFrame.from_dict(data_dict,
                            orient="index", columns=["one", "two", "three"])
print("data_dict")
print(data_dict, end="\n\n")
print("df_o")
print(df_o, end="\n\n")
print("df")
print(df, end="\n\n")

print(df["one"], end="\n\n")
print(df["two"], end="\n\n")
df["fourth"] = df["one"]
print(df, end="\n\n")

del df["one"]
df.pop("two")
print(df, end="\n\n")

df["random"] = "hello"
df["cut"] = df["three"][:1]
df.insert(2, "whatup", [100, 100])
print(df, end="\n\n")

ser = pd.Series([1, 2, 3], index=["a", "b", "c"])
print("ser.drop(labels=[\"b\"])")
print(ser.drop(labels=["b"]), end="\n\n")

print("df")
print(df, end="\n\n")

print(df.loc["A"], end="\n\n")
print(df.iloc[0], end="\n\n")

df_1 = pd.DataFrame(np.arange(0, 20).reshape(5, 4), columns=["A", "B", "C", "D"])
df_2 = pd.DataFrame(np.ones(9).reshape(3, 3), columns=["A", "B", "C"])
print("df_1")
print(df_1, end="\n\n")
print("df_2")
print(df_2, end="\n\n")
print("df_1.add(df_2, fill_value=None)")
print(df_1.add(df_2, fill_value=None), end="\n\n")
print("df_2 * 3 + 2")
print(df_2 * 3 + 2, end="\n\n")

print("df_1")
print(df_1, end="\n\n")
print(df_1.T, end="\n\n")

ser = pd.Series(np.random.randn(1000))
print("ser")
print(ser.head(), end="\n\n")
print(ser.tail(10), end="\n\n")

date_ind = pd.date_range('10/5/2022', periods=5)

df = pd.DataFrame(np.random.randn(5, 3), index=date_ind, columns=["A", "B", "C"])
print("df")
print(df, end="\n\n")

df = pd.DataFrame(np.random.randn(3, 4), index=["a", "b", "c"], columns=["A", "B", "C", "D"])

print(df.loc["c"], end="\n\n")
print(df.iloc[2], end="\n\n")
row_c = df.iloc[2]
print(df["A"], end="\n\n")
col_A = df["A"]

print(df.sub(row_c, axis=1), end="\n\n")
print(df.sub(col_A, axis=0), end="\n\n")

df = pd.DataFrame(np.arange(0, 12).reshape(4, 3))
df1 = pd.DataFrame(np.zeros(16).reshape(4, 4))
df = df + df1
print(df, end="\n\n")
print(df.mean(0), end="\n\n")
print(df.mean(1), end="\n\n")

print(df.sum(0, skipna=False), end="\n\n")
print(df.sum(1, skipna=True), end="\n\n")

df = pd.DataFrame(np.arange(0, 12).reshape(4, 3))
print(df.std(0), end="\n\n")
print(df.std(1), end="\n\n")

print(df.cumsum(0), end="\n\n")
print(df.cumsum(1), end="\n\n")

ser = pd.Series(np.random.randn(500))
ser[20:500] = np.nan
ser[10:20] = 5
print("ser.nunique()")
print(ser.nunique(), end="\n\n")

ser = pd.Series(np.random.randn(1000))
ser[::2] = np.nan
print(ser.idxmin(), end="\n\n")
print(ser.idxmax(), end="\n\n")
print(ser.describe(percentiles=[0.05, 0.10, 0.90]), end="\n\n")

ser = pd.Series(['a', 'a', 'c', 'c', 'b', np.nan, 'd', 'g'])
print(ser.describe(), end="\n\n")

df = pd.DataFrame({"x": ['a', 'a', 'c', 'c', 'b'], "y": np.random.random(5)})
print("df")
print(df.describe(include="all"), end="\n\n")

print(df["y"].idxmin(), end="\n\n")
print(df["y"].idxmax(), end="\n\n")
df = pd.DataFrame(np.arange(0, 12).reshape(4, 3))
print(df.apply(np.mean), end="\n\n")

df = pd.DataFrame({"A": np.arange(-3, 2), "B": np.ones(5)})
print(df, end="\n\n")
df = df.transform({"A": np.abs, "B": lambda x: x + 1})
print(df, end="\n\n")
