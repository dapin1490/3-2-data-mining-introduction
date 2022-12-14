from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 221214

data = pd.read_csv(r"weeks\week15\data\subway_1.csv", encoding="euc-kr")
# print(data.columns)  # ['사용일자', '노선명', '역명', '승차총승객수', '하차총승객수', '등록일자']

# 3-1 호선 분류하기
data_31 = data[['노선명', '승차총승객수', '하차총승객수']]
x = data_31[['승차총승객수', '하차총승객수']]
y = data['노선명']
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)

logreg = LogisticRegression()
logreg.fit(x_tr, y_tr)

# 입력받기
x1 = int(input('승차총승객수 : '))
x2 = int(input('하차총승객수 : '))
test_x = pd.DataFrame({'승차총승객수':[x1], '하차총승객수':[x2]})
test_y = logreg.predict(test_x)
print(f"입력 후 예측 : {test_y[0]}")

# 3-2 입력이 10000, 10000일 때
x1 = 10000
x2 = 10000
test_x = pd.DataFrame({'승차총승객수':[x1], '하차총승객수':[x2]})
test_y = logreg.predict(test_x)
print(f"10000, 10000 예측 : {test_y[0]}")
