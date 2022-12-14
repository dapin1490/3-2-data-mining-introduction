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
