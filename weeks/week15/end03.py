import pandas as pd
# from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"weeks\week15\data\London.csv", index_col=0).pivot_table(['Price', 'Area in sq ft'], 'House Type')
print(data)
sns.regplot(x='Price', y='Area in sq ft', data=data)
plt.show()
