"""
5개 columns에 숫자 200개씩 담은 csv가 존재한다. 해당 파일을 기반으로 numpy, pandas 각각 활용하여서 통계적인 분석 값들을 도출하여 출력하시오.
→ 주요 분석 값: 평균, 표준편차, 최고, 최저, 5분위(15%,35%,50%,65%,85%)
→ csv파일을 읽는 방식 제한 없음
→ 샘플개수 지정방식 제한 없음
→ 다만 판다스의 describe()를 포함한 기본적인 종합 통계값 계산 함수/메소드 사용 금지. (다만 자체적으로 만든 함수 및 메소드는 100% 사용 가능)
"""
# 과제 해설 : 넘파이와 판다스를 각각 써서 같은 결과를 두 번 출력하라. 다른 값은 같지만 std 값이 다르게 나오는데 이건 스스로 생각해 볼 것.

import pandas as pd
import numpy as np

def sample_mean(sam, mean):
	cols = list(sam.columns)
	for i in range(5):
		mean[i] = sam[cols[i]].sum() / sam[cols[i]].count()
	return

def sample_std(sam, mean, std_arr):
	cols = list(sam.columns)
	for i in range(5):
		sam[str(i + 5)] = (sam[cols[i]] - mean[i]) ** 2
		std_arr[i] = (sam[str(i + 5)].sum() / sam[str(i + 5)].count()) ** 0.5
	return

def sample_max_min(sam, maxs, mins):
	cols = list(sam.columns)
	for i in range(5):
		maxs[i] = sam[cols[i]].max()
		mins[i] = sam[cols[i]].min()
	return

def sample_top5(sam, tn, ts):
	cols = list(sam.columns)
	for i in range(5):
		sam = sam.sort_values(cols[i])
		for j in range(5):
			idx = int(sam[cols[i]].count() * 0.01 * tn[j])
			ts[j][i] = sam.iloc[(idx - 1):idx, i:(i + 1)].values
	return


data_csv = "weeks/week06/homework/testdata.csv"
tops = [15, 35, 50, 65, 85]
data = pd.read_csv(data_csv, index_col=0)
colum = list(data.columns)

means = np.zeros(5)  # 평균
stds = np.zeros(5)  # 표준편차 : 편차의 제곱의 평균의 제곱근
maxs = np.zeros(5)  # 최댓값
mins = np.zeros(5)  # 최솟값
top5 = np.zeros((5, 5))  # 5분위

sample_mean(data, means)
sample_max_min(data, maxs, mins)
sample_top5(data, tops, top5)
sample_std(data, means, stds)

results = pd.DataFrame({"mean": means, "std": stds, "max": maxs, "min": mins, "top 15%": top5[0], "top 35%": top5[1], "top 50%": top5[2], "top 65%": top5[3], "top 85%": top5[4]}, index=colum)
print("mean: 평균, std: 표준편차, max: 최고, min: 최소, [top 15%, top 35%, top 50%, top 65%, top 85%]: 5분위")
print(results.transpose())
