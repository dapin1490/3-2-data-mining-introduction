import matplotlib.pyplot as plt
from matplotlib import rc, font_manager  # 한글 출력하기
import numpy as np
import os  # 코드 실행 결과 저장할 폴더 있는지 확인하고, 없으면 만들기 위해 import

# matplotlib 그래프에 한글 출력하기 : 안 하면 네모 박스로 나옴
# 교수님이 써주신 코드이니 이런 게 있다는 사실은 기억해두는 게 좋다.
font_path = "c:/Windows/Fonts/HANDotum.ttf"  # 함초롬돋움
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 코드 실행 결과가 저장될 폴더가 없다면 하나 만들어준다
# 수업시간에 배운 내용 아님
# 파이참으로 실행하는 것을 전제로 하는 코드이며,
# 다른 프로그램으로 실행할 경우 의도하지 않은 곳에 폴더가 생성될 수 있으니
# 확인 후 경로를 수정해도 좋고(이 코드만 따로 떼서 실행하면 빈 폴더가 생성되기 때문에 잘못 만들었다면 바로 지우면 된다)
# 파이참으로 바로 실행해도 된다.
if not os.path.exists("images/"):
	os.mkdir("images/")

"""
Matplotlib 이미지 저장하기 참고 : https://codetorial.net/matplotlib/savefig.html
수업시간에 배운 거 아니고 보기 좋게 저장하기 위해 찾아본 것이니 신경쓰지 않아도 무방
각 예제는 01번부터 50번까지 총 51개의 이미지로 저장된다. 07번이 2개 있다.
한번에 모두 실행하면 시간이 오래 걸리므로 나눠서 실행해도 된다.
이미지를 저장하지 않으려면 savefig와 clf가 있는 줄을 주석처리하고
	show를 주석 해제하면 각 실행 결과가 일일이 창으로 뜬다.
savefig 이후에 clf를 쓰는 이유는 savefig만으로는 먼저 그려두었던 figure가 사라지지 않아서
	뒤로 갈수록 그래프가 합쳐져 원치 않는 결과가 되기 때문이다.
	clf는 현재 그려둔 figure를 지우는 코드이다.
"""

"""
수업 내내 데이터 시각화의 중요성을 강조하셨다.
어디 가서 발표할 일이 있을 때 열 줄 글보다 한 장의 그래프가 훨씬 보기 좋다며
	필요할 때 여기서 찾아 쓰라고 예제를 자세히 써주셨다.
"""

y = (16, 9, 4, 1, 0, 1, 4, 9, 16)
plt.plot(y, 'b')
plt.title('파형')
plt.ylabel('출력')
plt.xlabel('입력')
# plt.show()
plt.savefig(r"images/ex01.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

plt.plot([1, 3, 7, 11, 14, 15])
plt.xlabel('time[min]')
plt.ylabel('distance[km]')
# plt.show()
plt.savefig(r"images/ex02.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

plt.plot([1, 2, 3, 4, 5], [120, 200, 160, 350, 110])
plt.xlabel('input')
plt.ylabel('output')
plt.title('Experiment Result')
# plt.show()
plt.savefig(r"images/ex03.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

plt.title("plot subject", loc='right', fontdict={'fontsize': 10})  # title
plt.plot([10, 20, 30, 40], [1, 4, 9, 16], 'rs--')
plt.xlabel("time", fontdict={'fontsize': 10})  # x-axis label
plt.ylabel("amplitude", fontdict={'fontsize': 10})  # y-axis label
# plt.xlim(0, 50)
# plt.ylim(0, 20)
# plt.show()
plt.savefig(r"images/ex04.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

x = range(0, 5)
y = [v ** 2 for v in x]
s = [10000, 200, 300, 400, 500]
plt.scatter(x=x, y=y, s=s, c='blue', alpha=0.9)
# plt.show()
plt.savefig(r"images/ex05.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

plt.plot([1, 2, 3, 4], [1, 4, 9, 10], 'rs--')
plt.plot([2, 3, 4, 5], [5, 6, 7, 9], 'b^-')
plt.xlabel('input')
plt.ylabel('output)')
plt.title('Experiment Result')
plt.legend(['Dog', 'Cat'], loc='lower right')
# plt.show()
plt.savefig(r"images/ex06.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

x = range(0, 100)
y1 = [v ** 2 for v in x]
y2 = [v ** 3 for v in x]
plt.subplot(2, 1, 1)  # row의 개수, column의 개수, 몇 번째
plt.ylabel('v^2')
plt.plot(x, y1)
plt.subplot(2, 1, 2)
plt.ylabel('v^3')
plt.plot(x, y2)
# plt.show()
plt.savefig(r"images/ex07(1).png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

x = range(0, 100)
y1 = [v ** 2 for v in x]
y2 = [v ** 3 for v in x]
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)  # row의 개수, column의 개수, 몇 번째
plt.ylabel('v^2')
plt.plot(x, y1)
plt.grid()
plt.subplot(1, 2, 2)
plt.ylabel('v^3')
plt.plot(x, y2)
# plt.show()
plt.savefig(r"images/ex07(2).png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

x = range(0, 20)
y = [v ** 2 for v in x]
plt.bar(x, y, width=0.5, color="red")
# plt.show()
plt.savefig(r"images/ex08.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

# tick locations and labels
x = range(0, 6)
y = [4000, 5000, 3000, 2000, 4500, 5500]
plt.xticks(np.arange(6), ('아빠는 외계인', '민초', '레샤', '바닐라', '쿠키앤도우', '슈팅스타'))
plt.bar(x, y, width=0.5, color="red")
# plt.show()
plt.savefig(r"images/ex09.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

plt.xticks(np.arange(6), ('아빠는 외계인', '민초', '레샤', '바닐라', '쿠키앤도우', '슈팅스타'))
plt.yticks([2.5, 7.5, 12.5, 17.5, 22.5, 27.5], ('만족도', '품질', '고객경험', '주요 타겟', '주문량', '공급량'))

# ticks, tick labels, and gridlines
plt.tick_params(axis='x', labelsize=13, colors='b')
plt.tick_params(axis='y', labelsize=13, colors='r')
plt.grid()
# plt.show()
plt.savefig(r"images/ex10.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

plt.figure(figsize=(10, 2))
x = np.linspace(0, 1.0)
y = np.cos(np.pi * 2 * x) ** 2
plt.plot(x, y)
# plt.show()
plt.savefig(r"images/ex11.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

plt.figure(figsize=(10, 2))
x = np.linspace(0, 1.0)
y = np.cos(np.pi * 2 * x) ** 2
plt.plot(x, y, marker='x', markersize=8, linestyle=':', linewidth=3, color='g', label='$\cos^2(\pi x)$')  # 라벨은 LaTex 서식으로 쓰임
plt.legend(loc='lower right')
plt.xlabel('input')
plt.ylabel('output')
plt.title('Experiment Result')
# plt.show()
plt.savefig(r"images/ex12.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

plt.figure(figsize=(10, 2))
x = np.linspace(0, 1.0)
y = np.cos(np.pi * 2 * x) ** 2
plt.plot(x, y, marker='x', markersize=8, linestyle=':', linewidth=3, color='g', label='$\sin^2(\pi x)$')
plt.legend(loc='lower right')
plt.xlabel('input')
plt.ylabel('output')
plt.title('Experiment Result')
# plt.savefig('func_plot.jpg')  # 이 예제는 원래 savefig 실습하는 예제였다. 파일명이 달라지면 이미지 정리에 방해돼서 실행하지 않았음.
plt.savefig(r"images/ex13.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

print("-----------------------")
plt.figure()
plt.axes()
circle = plt.Circle((-1, -1), 1, fc='g', ec="red")
plt.gca().add_patch(circle)
rectangle = plt.Rectangle((0, 0), 4, 2, fc='b', ec="red")
plt.gca().add_patch(rectangle)
rectangle = plt.Rectangle((-5, -5), 3, 3, fc='y', ec="red")
plt.gca().add_patch(rectangle)
plt.axis('scaled')
plt.grid(linestyle='--')
# plt.show()
plt.savefig(r"images/ex14.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

circle1 = plt.Circle((-1, -1), 0.8, color='g')
circle2 = plt.Circle((0, 0), 0.4, color='b')
circle3 = plt.Circle((1, 1), 0.2, color='r', clip_on=True)

# clipped circle(clip_on=True(default))
fig, ax = plt.subplots()  # fig = plt.gcf(), ax = fig.gca()
plt.xlim(-1, 1)  # change x-range default
plt.ylim(-1, 1)  # change y-range default
plt.grid(linestyle='--')
ax.set_aspect(1)
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
plt.title('plot a circle with matplotlib', fontsize=10)
# plt.show()
plt.savefig(r"images/ex15.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
# 16번부터 18번 예제는 3D 예제이므로 가능하다면 꼭 한번 plt.show()로 실행해보기 바란다.
# 저장된 이미지는 상호작용할 수 없지만 plt.show()로 뜨는 이미지는 이리저리 돌려볼 수 있다.

fig = plt.figure(figsize=(10, 5))
axis = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

import numpy as np

t = np.linspace(0.0, 5.0, 500)
x = np.cos(np.pi * t)
y = np.sin(np.pi * t)
z = 2 * t
axis.plot(x, y, z)
axis.set_xlabel('x-axis')
axis.set_ylabel('y-axis')
axis.set_zlabel('z-axis')
# plt.show()
plt.savefig(r"images/ex16.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

fig = plt.figure(figsize=(10, 5))
axis = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
x = np.linspace(0.0, 1.0)  # x, y are vectors
y = np.linspace(0.0, 1.0)
X, Y = np.meshgrid(x, y)  # X, Y are 2d arrays
Z = X ** 2 - Y ** 2  # points in the z axis
axis.plot_surface(X, Y, Z)  # data values (2D Arryas)
axis.set_xlabel('x-axis')
axis.set_ylabel('y-axis')
axis.set_zlabel('z-axis')
axis.view_init(elev=30, azim=70)  # elevation & angle
axis.dist = 10  # distance from the plot
# plt.show()
plt.savefig(r"images/ex17.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

from matplotlib import cm

fig = plt.figure(figsize=(10, 5))
axis = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
p = axis.plot_surface(X, Y, Z, rstride=1, cstride=2, cmap=cm.coolwarm, linewidth=1)
axis.set_xlabel('x-axis')
axis.set_ylabel('y-axis')
axis.set_zlabel('z-axis')
axis.view_init(elev=30, azim=70)  # elevation & angle
fig.colorbar(p, shrink=0.5)
# plt.show()
plt.savefig(r"images/ex18.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2020', periods=1000))
ser = s.cumsum()
roll = s.rolling(window=60)

ser.plot(style='k')
# plt.show()
plt.savefig(r"images/ex19.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

roll.mean().plot(style='k--')
# plt.show()
plt.savefig(r"images/ex20.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df = pd.DataFrame(np.random.randn(1000, 3), index=pd.date_range('1/1/2020', periods=1000), columns=['A', 'B', 'C'])
dfc = df.cumsum()
dfc.rolling(window=60).sum().plot(subplots=True)
# plt.show()
plt.savefig(r"images/ex21.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

s = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2020', periods=1000))

ser = s.cumsum()

rol = ser.rolling(window=60)
rol.mean().plot(style='k--')

ser.expanding().mean().plot(style='k')
# plt.show()
plt.savefig(r"images/ex22.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

ser.ewm(span=20).mean().plot(style='k')
rol.mean().plot(style='k--')
# plt.show()
plt.savefig(r"images/ex23.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

print("------------------------------------")
df1 = pd.DataFrame(np.random.rand(7, 4), columns=['a', 'b', 'c', 'd'])

import matplotlib.pyplot as plt

plt.close('all')
plt.figure()
df1.iloc[3].plot(kind='bar')
# plt.show()
plt.savefig(r"images/ex24.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

plt.figure()
df1.iloc[3].plot.bar()
plt.axhline(0, color='b')
# plt.show()
plt.savefig(r"images/ex25.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df2 = pd.DataFrame(np.random.rand(7, 3), columns=['a', 'b', 'c'])
df2.plot.bar()
# plt.show()
plt.savefig(r"images/ex26.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df2.plot.bar(stacked=True)
# plt.show()
plt.savefig(r"images/ex27.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df2.plot.barh(stacked=True)
# plt.show()
plt.savefig(r"images/ex28.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df3 = pd.DataFrame({'a': np.random.randn(500) + 1, 'b': np.random.randn(500), 'c': np.random.randn(500) - 1}, columns=['a', 'b', 'c'])

df3.plot.hist(alpha=0.5)
# plt.show()
plt.savefig(r"images/ex29.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df3.plot.hist(stacked=True, bins=12)
# plt.show()
plt.savefig(r"images/ex30.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df3['a'].plot.hist(orientation='horizontal', cumulative=True)
# plt.show()
plt.savefig(r"images/ex31.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

ser = pd.Series(np.random.randn(500))
ser.hist(by=np.random.randint(0, 4, 500), figsize=(7, 4))
# plt.show()
plt.savefig(r"images/ex32.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df = pd.DataFrame(np.random.rand(5, 4), columns=['A', 'B', 'C', 'D'])
df.plot.box()
# plt.show()
plt.savefig(r"images/ex33.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame(np.random.rand(5, 4), columns=['A', 'B', 'C', 'D'])
df.plot.box()
# plt.show()
plt.savefig(r"images/ex34.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

color = {'boxes': 'blue', 'whiskers': 'red', 'medians': 'green', 'caps': 'orange'}
df.plot.box(color=color, sym='r+')
# plt.show()
plt.savefig(r"images/ex35.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df.plot.box(vert=False, positions=[1, 3, 4, 5])
# plt.show()
plt.savefig(r"images/ex36.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

bp = df.boxplot()
# plt.show()
plt.savefig(r"images/ex37.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df = pd.DataFrame(np.random.rand(10, 2), columns=['HA', 'HI'])
df['HO'] = pd.Series(['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'])
bp = df.boxplot(by='HO')
# plt.show()
plt.savefig(r"images/ex38.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df1 = pd.DataFrame(np.random.rand(7, 3), columns=['HA', 'HI', 'HO'])
df1['X'] = pd.Series(['A', 'A', 'A', 'A', 'B', 'B', 'B'])
df1['Y'] = pd.Series(['A', 'B', 'A', 'B', 'A', 'B', 'A'])

bp = df1.boxplot(column=['HA', 'HO'], by=['X', 'Y'])
# plt.show()
plt.savefig(r"images/ex39.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df = pd.DataFrame(np.random.rand(5, 3), columns=['HA', 'HI', 'HO'])
df.plot.area()
# plt.show()
plt.savefig(r"images/ex40.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df.plot.area(stacked=False)
# plt.show()
plt.savefig(r"images/ex41.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df = pd.DataFrame(np.random.rand(20, 4), columns=['a', 'b', 'c', 'd'])

df.plot.scatter(x='a', y='b')
# plt.show()
plt.savefig(r"images/ex42.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

axsub = df.plot.scatter(x='a', y='b', color='darkgreen', label='Group A')
df.plot.scatter(x='c', y='d', color='red', label='Group B', ax=axsub)
# plt.show()
plt.savefig(r"images/ex43.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df.plot.scatter(x='a', y='b', c='c', s=100)
# plt.show()
plt.savefig(r"images/ex44.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df.plot.scatter(x='a', y='b', s=df['d'] * 500)
# plt.show()
plt.savefig(r"images/ex45.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df = pd.DataFrame(np.random.randn(1000, 2), columns=['HA', 'HI'])
df['HI'] = df['HI'] + np.arange(1000)
df.plot.hexbin(x='HA', y='HI', gridsize=20)
# plt.show()
plt.savefig(r"images/ex46.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df['HO'] = np.random.uniform(0, 3, 1000)
df.plot.hexbin(x='HA', y='HI', C='HO', reduce_C_function=np.max, gridsize=20)
# plt.show()
plt.savefig(r"images/ex47.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

ser = pd.Series(np.random.rand(4), index=['A', 'B', 'C', 'D'], name='series')
ser.plot.pie(figsize=(5, 5))
# plt.show()
plt.savefig(r"images/ex48.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

df = pd.DataFrame(np.random.rand(3, 2), index=['A', 'B', 'C'], columns=['HA', 'HO'])
df.plot.pie(subplots=True, figsize=(10, 5))
# plt.show()
plt.savefig(r"images/ex49.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()

ser.plot.pie(labels=['A', 'B', 'C', 'D'], colors=['r', 'g', 'b', 'y'], autopct='%.2f', fontsize=15, figsize=(5, 5))

ser = pd.Series([0.24] * 4, index=['A', 'B', 'C', 'D'], name='series1')
ser.plot.pie(figsize=(5, 5))
# plt.show()
plt.savefig(r"images/ex50.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
