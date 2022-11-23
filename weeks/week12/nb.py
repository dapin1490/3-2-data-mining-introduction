# 0. 라이브러리
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import time
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager  # 한글 출력하기

font_path = "c:/Windows/Fonts/HANDotum.ttf"  # 함초롬돋움
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

res = open("res/nb_result.txt", 'w')

res.write(f"nb.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}\n\n")
print(f"nb.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}")

res.write("나이브 베이즈 실습\n")
res.write("나이브 베이즈가 종류가 많은데 대체로 가우시안 나이브 베이즈 쓴다고 알고 있으면 된다.\n\n")

# 1. 데이터 로드
X, y = load_iris(return_X_y=True)

# 2. 학습셋 테스트셋 분리
# train_test_split 중요
# 보통 학습셋은 8:2 정도로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 3. 가우시안 나이브 베이즈 모델 준비
gnb = GaussianNB()

# 4. 모델 학습
# y_pred = gnb.fit(X_train, y_train).predict(X_test)
gnb.fit(X_train, y_train)

# 5. 모델 테스트
y_pred = gnb.predict(X_test)

# 6. 예측
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

res.write(f"Number of mislabeled points out of a total {X_test.shape[0]} points : {(y_test != y_pred).sum()}\n")

res.write("\n-----\n\n번외편\n\n")

i_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
scores = []

for i in i_list:
	X, y = load_iris(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=0)
	gnb = GaussianNB()
	gnb.fit(X_train, y_train)
	y_pred = gnb.predict(X_test)
	res.write(f"test_size = {i}\n")
	res.write(f"Number of mislabeled points out of a total {X_test.shape[0]} points : {(y_test != y_pred).sum()}\n")
	res.write(f"correct rate : {(X_test.shape[0] - (y_test != y_pred).sum()) / X_test.shape[0] :.2f}\n\n")
	scores.append((X_test.shape[0] - (y_test != y_pred).sum()) / X_test.shape[0])

plt.bar([str(i) for i in i_list], scores, alpha=0.7, color='blue')
plt.title("테스트셋 분할 비율에 따른 성능")
plt.xlabel("테스트셋 비율")
plt.ylabel("정확도")
plt.ylim(0.9, 1.01)
# plt.show()
plt.savefig(r"res/nb_image.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("res/nb_image.png")

res.close()
