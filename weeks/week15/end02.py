import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"weeks\week15\data\airtravel.csv")

heat = data.pivot_table([' "1958"', ' "1959"', ' "1960"'], 'Month')
sns.heatmap(heat)
plt.show()
