"""
1. "seoul_ems.csv" 파일이 존재한다. 해당 파일을 읽고 항목별 홀수년도 데이터만 존재하는 행렬로 데이터 처리를 하시요. 다만 하기 조건을 따르시요. (7점)
- Null값의 경우 0으로 대체
- 총계 삭제 필요
- 10개 미만의 데이터 경우 10개로 재구성
"""

import pandas as pd

data = pd.read_csv(r'data/seoul_ems.csv', index_col=0)

# data.isnull().sum() == 0이므로 Null 없음
data.drop('총계', axis=1, inplace=True)  # 총계 삭제
# len(data.index) == 25이므로 10개 미만 데이터 아님

for i in data.columns:
	year = int(i)
	if year % 2 == 0:
		data.drop(i, axis=1, inplace=True)

print(data.info(), end="\n\n")
print(data)
data.to_csv(r'data/seoul_ems_answer.csv')
