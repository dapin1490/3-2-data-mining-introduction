import pandas as pd
import numpy as np

data = np.random.random(4)
index = ['a', 'b', 'c', 'd']
ser = pd.Series(data, index=index)
data = [1, 1, 1, 2, 3, 3, 4, 2, 3, 4]
ser1 = pd.Series(data)

print(ser)
print(ser1)

print(ser1.values)
print(ser1.index)
print(ser1.value_counts())

dic = {"kimbap": "5000", "Sundae": "4000"}
ser2 = pd.Series(dic)
print(ser2)
ser2 = pd.Series(dic, index=["a", "Sundae", "kimbap", "d"])
print(ser2)

ser3 = pd.Series(42, index=["Answer", "of", "the", "Universe"])
print(ser3)
print(ser2[0])
print(ser2[1])
print(ser2["a"])
print(ser2["Sundae"])

print(ser1[:4])
print(ser1[::2])
print(np.exp(ser1[::2]))
print(np.log(ser1[::2]))

ser4 = pd.Series(np.random.random(10))
ser5 = pd.Series(np.random.random(4))
ser4_1 = ser4[:4]
print(ser4_1 + ser5)

ser5_1 = ser5[1:] + ser5[:-1]
print(ser5_1)

ser6 = pd.Series(data, name="abcd")
print(ser6)
ser6_1 = ser6.rename("dcba")
print(ser6_1.name)

print("-----------------------------")

d = {"하나": [0, 1, 2, 4], "둘": [0, 1, 2, 3]}
df = pd.DataFrame(d)
print(df)

df = pd.DataFrame(d, index=["b", "c", "a", "a"], columns=["영", "둘"])
print(df)
print(df.index)
print(df.columns)

arr = np.zeros((2,), dtype=[('A', 'i4'), ('B', 'f4'), ('D', 'a10')])
arr[:] = [(1, 2, "Hello"), (2, 3, "World")]
df = pd.DataFrame(arr)
df1 = pd.DataFrame(arr, index=['first', 'second'])
df2 = pd.DataFrame(arr, columns=['C', 'A', 'B'])
print(df)
print(df1)
print(df2)

data_dict = dict([('A', [1, 2, 3]), ('B', [4, 5, 6])])
df_o = pd.DataFrame(data_dict)

df = pd.DataFrame.from_dict(data_dict,
                            orient="index", columns=["one", "two", "three"])
print(data_dict)
print(df_o)
print(df)

print(df["one"])
print(df["two"])
df["fourth"] = df["one"]
print(df)

del df["one"]
df.pop("two")
print(df)

df["random"] = "hello"
df["cut"] = df["three"][:1]
df.insert(2, "whatup", [100, 100])
print(df)

ser = pd.Series([1, 2, 3], index=["a", "b", "c"])
print(ser.drop(labels=["b"]))

print(df)

print(df.loc["A"])
print(df.iloc[0])

df_1 = pd.DataFrame(np.arange(0, 20).reshape(5, 4), columns=["A", "B", "C", "D"])
df_2 = pd.DataFrame(np.ones(9).reshape(3, 3), columns=["A", "B", "C"])
print(df_1)
print(df_2)
print(df_1.add(df_2, fill_value=None))
print(df_2 * 3 + 2)

print(df_1)
print(df_1.T)

ser = pd.Series(np.random.randn(1000))
print(ser.head())
print(ser.tail(10))

date_ind = pd.date_range('10/5/2022', periods=5)

df = pd.DataFrame(np.random.randn(5, 3), index=date_ind, columns=["A", "B", "C"])
print(df)

df = pd.DataFrame(np.random.randn(3, 4), index=["a", "b", "c"], columns=["A", "B", "C", "D"])

print(df.loc["c"])
print(df.iloc[2])
row_c = df.iloc[2]
print(df["A"])
col_A = df["A"]

print(df.sub(row_c, axis=1))
print(df.sub(col_A, axis=0))

df = pd.DataFrame(np.arange(0, 12).reshape(4, 3))
df1 = pd.DataFrame(np.zeros(16).reshape(4, 4))
df = df + df1
print(df)
print(df.mean(0))
print(df.mean(1))

print(df.sum(0, skipna=False))
print(df.sum(1, skipna=True))

df = pd.DataFrame(np.arange(0, 12).reshape(4, 3))
print(df.std(0))
print(df.std(1))

print(df.cumsum(0))
print(df.cumsum(1))

ser = pd.Series(np.random.randn(500))
ser[20:500] = np.nan
ser[10:20] = 5
print(ser.nunique())

ser = pd.Series(np.random.randn(1000))
ser[::2] = np.nan
print(ser.idxmin())
print(ser.idxmax())
print(ser.describe(percentiles=[0.05, 0.10, 0.90]))

ser = pd.Series(['a', 'a', 'c', 'c', 'b', np.nan, 'd', 'g'])
print(ser.describe())

df = pd.DataFrame({"x": ['a', 'a', 'c', 'c', 'b'], "y": np.random.random(5)})
print(df.describe(include="all"))

print(df["y"].idxmin())
print(df["y"].idxmax())
df = pd.DataFrame(np.arange(0, 12).reshape(4, 3))
print(df.apply(np.mean))

df = pd.DataFrame({"A": np.arange(-3, 2), "B": np.ones(5)})
print(df)
df = df.transform({"A": np.abs, "B": lambda x: x + 1})
print(df)
