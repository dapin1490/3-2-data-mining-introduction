"""
3. 하기와 같은 코드가 존재한다. 해당 코드 아랫부분에 추가로 9줄 이하로 작성하여 하기 그림과 동일한 출력을 생성하시오. (7점)
- 단 ;을 활용한 여러줄의 한줄 축약은 금지한다.

import pandas as pd
import numpy as np

df1_data = np.ones((5, 6))
df1 = pd.DataFrame(df1_data, index=list(range(0, 10, 2)), columns=list(range(0, 12, 2)))
df2_data = np.ones((6, 5))
df2 = pd.DataFrame(df2_data, index=list(range(0, 12, 2)), columns=list(range(0, 10, 2)))
df3 = df1 + df2

최종 출력:
     0    2    4    6    8   10
0   2.0  NaN  2.0  NaN  2.0 NaN
2   NaN  NaN  2.0  NaN  2.0 NaN
4   2.0  2.0  2.0  NaN  2.0 NaN
6   NaN  NaN  NaN  NaN  2.0 NaN
8   2.0  2.0  2.0  2.0  2.0 NaN
10  NaN  NaN  NaN  NaN  NaN NaN
"""

import pandas as pd
import numpy as np

df1_data = np.ones((5, 6))
df1 = pd.DataFrame(df1_data, index=list(range(0, 10, 2)), columns=list(range(0, 12, 2)))
df2_data = np.ones((6, 5))
df2 = pd.DataFrame(df2_data, index=list(range(0, 12, 2)), columns=list(range(0, 10, 2)))
df3 = df1 + df2

# write under 9 lines or same -------------
for i in range(0, 6):
	for j in range(0, 6):
		if (i == j and i % 2 != 0) or (i > j and i % 2 != 0) or (i < j and j % 2 != 0):
			df3.iloc[i, j] = np.nan
print(df3)
# ----------------
