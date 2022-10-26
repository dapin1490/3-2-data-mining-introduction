"""
5. "seoul_ems_test.csv"가 존재한다. 해당 csv의 데이터내 값에 대한 검증을 위해서 해당 데이터를 읽고 총계과 실제 데이터값의 합이 동일한지 여부에 대해서 파이썬 코드를 작성하여 검증을 진행하시요. 다시 말씀드리면 1)총계값과 2012년 이전~2017년도 데이터간의 합산 간 구별로 동일한지 틀린지 여부를 체크하고, 2) 틀리다면 총계값과 2012년 이전~2017년도 데이터 합산값간의 차이를 출력하시요. (7점)
"""

import pandas as pd

data = pd.read_csv(r'data/seoul_ems_test.csv', index_col=0)

print("총계값과 합산값이 같다면 True, 아니면 False와 (총계값 - 합산값) 출력")

for i in data.index:
	num_total = data.loc[i].iloc[len(data.loc[i]) - 1]
	sum_total = sum(data.loc[i].iloc[:len(data.loc[i]) - 1])
	print(f"{i} : {num_total == sum_total if num_total == sum_total else str(num_total == sum_total) + ', ' + str(num_total - sum_total)}")
