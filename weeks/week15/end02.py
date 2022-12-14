"""
1959~1961년까지의 항공편수가 기록된 airtravel.csv가 존재한다. 해당 csv를 활용하여서 년도와 월별로 하여서 항공편수의 동향을 파악할 수 있는 colormap 혹은 heatmap을 출력하시요. 다만 다음 조건을 따르시요. (5점)
- X, Y축 라벨이 정상적으로 입력되어야 한다.
(5점)
"""
# 해결함

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"weeks\week15\data\airtravel.csv")

heat = data.pivot_table([' "1958"', ' "1959"', ' "1960"'], 'Month')
sns.heatmap(heat)
plt.show()
