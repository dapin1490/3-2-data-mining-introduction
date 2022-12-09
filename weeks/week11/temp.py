import matplotlib.pyplot as plt
import seaborn as sns  # tips, dots, anscombe, fmri 데이터셋 원래 출처
import pandas as pd
import numpy as np
from matplotlib import rc, font_manager  # 한글 출력하기

# 히트맵 예제 링크 : https://seaborn.pydata.org/generated/seaborn.heatmap.html
# 산점도 검색
# K-최근접 이웃 알고리즘
# 회귀

# 지금 이 파일
# 기말 메모
# 4번째 과제 제출본
# 12주 필기

# matplotlib 그래프에 한글 출력하기 : 안 하면 네모 박스로 나옴
# 교수님이 써주신 코드이니 이런 게 있다는 사실은 기억해두는 게 좋다.
font_path = "c:/Windows/Fonts/HANDotum.ttf"  # 함초롬돋움
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# x = np.random.normal(size=100)
# sns.distplot(x)  # UserWarning: `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
# sns.histplot(x, kde=True)
# plt.xlabel("height")
# plt.ylabel("Counts")
# plt.show()

# data = sns.load_dataset('tips')
# # plt.figure(figsize=(13, 10))   # 그래프의 크기를 정합니다.
# # 그래프의 속성을 결정합니다. vmax의 값을 0.5로 지정해 0.5에 가까울 수록 밝은 색으로 표시되게 합니다.
# # sns.heatmap(data.corr(), linewidths=0.1, vmax=1, annot=True, cmap="YlGnBu")
# sns.heatmap(data.corr())
# plt.show()

data = pd.read_csv(r"weeks\week13+hw\homework\student_health_3.csv", encoding="euc-kr", header=0)
data = data[["몸무게", "키", "수축기", "이완기", "학년"]]
# print(data.info())
# print(data.corr())
# sns.heatmap(data.corr())
# plt.show()
# sns.distplot(data["키"].to_numpy())  # UserWarning: `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
# sns.histplot(data["키"].to_numpy(), kde=True)
# plt.xlabel("height")
# plt.ylabel("Counts")
# plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
# 16번부터 18번 예제는 3D 예제이므로 가능하다면 꼭 한번 plt.show()로 실행해보기 바란다.
# 저장된 이미지는 상호작용할 수 없지만 plt.show()로 뜨는 이미지는 이리저리 돌려볼 수 있다.

fig = plt.figure(figsize=(10, 5))
axis = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

import numpy as np

# t = np.linspace(0.0, 5.0, 500)
# x = np.cos(np.pi * t)
# y = np.sin(np.pi * t)
# z = 2 * t
# axis.plot(x, y, z)
# axis.set_xlabel('x-axis')
# axis.set_ylabel('y-axis')
# axis.set_zlabel('z-axis')
# plt.show()

x = data["몸무게"]
y = data["키"]
z = data["학년"]
axis.scatter(x, y, z)  # 산점도
axis.set_xlabel('x-몸무게')
axis.set_ylabel('y-키')
axis.set_zlabel('z-학년')
plt.show()

# fig = plt.figure(figsize=(10, 5))
# axis = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
# x = np.linspace(0.0, 1.0)  # x, y are vectors
# y = np.linspace(0.0, 1.0)
# X, Y = np.meshgrid(x, y)  # X, Y are 2d arrays
# Z = X ** 2 - Y ** 2  # points in the z axis
# axis.plot_surface(X, Y, Z)  # data values (2D Arryas)
# axis.set_xlabel('x-axis')
# axis.set_ylabel('y-axis')
# axis.set_zlabel('z-axis')
# axis.view_init(elev=30, azim=70)  # elevation & angle
# axis.dist = 10  # distance from the plot
# plt.show()

# from matplotlib import cm

# fig = plt.figure(figsize=(10, 5))
# axis = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
# p = axis.plot_surface(X, Y, Z, rstride=1, cstride=2, cmap=cm.coolwarm, linewidth=1)
# axis.set_xlabel('x-axis')
# axis.set_ylabel('y-axis')
# axis.set_zlabel('z-axis')
# axis.view_init(elev=30, azim=70)  # elevation & angle
# fig.colorbar(p, shrink=0.5)
# plt.show()

"""
model = Sequential()
model.add(Dense(1024, input_dim=input_size, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dense(num_classes, activation='softmax'))
"""
