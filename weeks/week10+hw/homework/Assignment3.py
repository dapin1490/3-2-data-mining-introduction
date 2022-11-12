"""
student_health_2.csv 데이터가 존재한다.
1) 학년별 학생 수 카운트를 진행하고,
2) 학년별 평균 키, 몸무게, 수축기, 이완기를 계산하여 Bar 형태로 출력하시요. (5점)
→ 총 5개의 plot이 생성되어야 함
→ pandas와 numpy 활용 필수
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rc  # 한글 출력하기

# 한글 출력하기
font_path = "c:/Windows/Fonts/HANDotum.ttf"  # 함초롬돋움
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 파일 불러오기
folder_route = r"weeks\week10+hw\homework\data"
file_route = folder_route + r"\student_health_2.csv"
rawdata = pd.read_csv(file_route, header=0, encoding='euc-kr')  # 참고 : https://teddylee777.github.io/pandas/%EA%B3%B5%EA%B3%B5%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%95%9C%EA%B8%80%EA%B9%A8%EC%A7%90%ED%98%84%EC%83%81-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95

# 1) 학년별 학생 수 카운트
grade_count = rawdata.groupby('학년')['ID'].count()
x = list(grade_count.index)
y = list(grade_count.values)

plt.bar(x, y)
plt.grid()
plt.xlabel('학년', fontdict={'fontsize': 15})
plt.ylabel('학생 수', fontdict={'fontsize': 15})
plt.title('학년별 학생 수', fontdict={'fontsize': 20})
plt.show()
# plt.savefig(folder_route + r"\answer01.png", facecolor='#dddddd', bbox_inches='tight')
# plt.clf()

# 2) 학년별 평균 키, 몸무게, 수축기, 이완기를 계산하여 Bar 형태로 출력하시요. (5점)
"""
column 이름 : '키', '몸무게', '수축기', '이완기'
"""
by_grade = rawdata.groupby('학년')[['키', '몸무게', '수축기', '이완기']].mean()  # <class 'pandas.core.frame.DataFrame'>
x = list(by_grade.index)

## 평균 키
height_mean = list(by_grade['키'])

plt.bar(x, height_mean)
plt.grid()
plt.xlabel('학년', fontdict={'fontsize': 15})
plt.ylabel('평균 키', fontdict={'fontsize': 15})
plt.title('학년별 평균 키', fontdict={'fontsize': 20})
plt.show()
# plt.savefig(folder_route + r"\answer02.png", facecolor='#dddddd', bbox_inches='tight')
# plt.clf()

## 평균 몸무게
weight_mean = list(by_grade['몸무게'])

plt.bar(x, weight_mean)
plt.grid()
plt.xlabel('학년', fontdict={'fontsize': 15})
plt.ylabel('평균 몸무게', fontdict={'fontsize': 15})
plt.title('학년별 평균 몸무게', fontdict={'fontsize': 20})
plt.show()
# plt.savefig(folder_route + r"\answer03.png", facecolor='#dddddd', bbox_inches='tight')
# plt.clf()

## 평균 수축기
systolic_mean = list(by_grade['수축기'])

plt.bar(x, systolic_mean)
plt.grid()
plt.xlabel('학년', fontdict={'fontsize': 15})
plt.ylabel('평균 수축기', fontdict={'fontsize': 15})
plt.title('학년별 평균 수축기', fontdict={'fontsize': 20})
plt.show()
# plt.savefig(folder_route + r"\answer04.png", facecolor='#dddddd', bbox_inches='tight')
# plt.clf()

## 평균 이완기
diastolic_mean = list(by_grade['이완기'])

plt.bar(x, diastolic_mean)
plt.grid()
plt.xlabel('학년', fontdict={'fontsize': 15})
plt.ylabel('평균 이완기', fontdict={'fontsize': 15})
plt.title('학년별 평균 이완기', fontdict={'fontsize': 20})
plt.show()
# plt.savefig(folder_route + r"\answer05.png", facecolor='#dddddd', bbox_inches='tight')
# plt.clf()
