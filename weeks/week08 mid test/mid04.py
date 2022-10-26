"""
4. 하기와 같은 랜덤숫자들이 존재한다. 1) 하기 랜덤숫자의 평균, 분산, 표준편차, 중위값,하위 20%, 80%값을 numpy와 pandas로 각각 구하고 2) 개별 값의 numpy와 pandas 값의 차이를 계산해서 출력하시요. 다만 하기 조건들을 따르시오. (7점)
- 1)과 2) 출력시 numpy, pandas 적용 여부 출력 동반 필요
- pandas의 describe 사용 금지

41, 195, 23, 111, 130, 120, 76, 117, 178, 113, 74, 186, 144, 92, 74, 15, 14, 36, 71, 171, 156, 108, 32, 143, 65, 35, 58, 54, 153, 21, 108, 67, 67, 9, 89, 112, 84, 77, 37, 137, 94, 18, 65, 155, 174, 54, 62, 190, 121, 84, 6, 194, 55, 43, 22, 121, 178, 49, 23, 169, 88, 113, 53, 110, 11, 15, 197, 103, 119, 67, 90, 135, 152, 128, 0, 90, 54, 174, 69, 162, 169, 84, 190, 94, 97, 77, 190, 158, 31, 17, 152, 12, 177, 35, 1, 117, 52, 185, 51, 95
"""

import numpy as np
import pandas as pd

data = [41, 195, 23, 111, 130, 120, 76, 117, 178, 113, 74, 186, 144, 92, 74, 15, 14, 36, 71, 171, 156, 108, 32, 143, 65, 35, 58, 54, 153, 21, 108, 67, 67, 9, 89, 112, 84, 77, 37, 137, 94, 18, 65, 155, 174, 54, 62, 190, 121, 84, 6, 194, 55, 43, 22, 121, 178, 49, 23, 169, 88, 113, 53, 110, 11, 15, 197, 103, 119, 67, 90, 135, 152, 128, 0, 90, 54, 174, 69, 162, 169, 84, 190, 94, 97, 77, 190, 158, 31, 17, 152, 12, 177, 35, 1, 117, 52, 185, 51, 95]
np_data = np.array(data)
pd_data = pd.Series(data)

np_list = [np_data.mean(), np_data.std(), np_data.var(), np.median(np_data), np.quantile(np_data, 0.2), np.quantile(np_data, 0.8)]
pd_list = [pd_data.mean(), pd_data.std(), pd_data.var(), pd_data.median(), pd_data.quantile(0.2), pd_data.quantile(0.8)]

print("1) 각종 통계값 출력")
print("---\nnumpy\n---")
print(f"평균 : {np_list[0]}")
print(f"분산 : {np_list[1]}")
print(f"표준편차 : {np_list[2]}")
print(f"중위값 : {np_list[3]}")
print(f"하위 20% : {np_list[4]}")
print(f"하위 80% : {np_list[5]}\n---")

print()

print("---\npandas\n---")
print(f"평균 : {pd_list[0]}")
print(f"분산 : {pd_list[1]}")
print(f"표준편차 : {pd_list[2]}")
print(f"중위값 : {pd_list[3]}")
print(f"하위 20% : {pd_list[4]}")
print(f"하위 80% : {pd_list[5]}\n---")

print()

print("2) numpy와 pandas 통계값 차이 출력")
print("---\n(numpy - pandas)\n---")
print(f"평균 : {np_list[0] - pd_list[0]}")
print(f"분산 : {np_list[1] - pd_list[1]}")
print(f"표준편차 : {np_list[2] - pd_list[2]}")
print(f"중위값 : {np_list[3] - pd_list[3]}")
print(f"하위 20% : {np_list[4] - pd_list[4]}")
print(f"하위 80% : {np_list[5] - pd_list[5]}\n---")
