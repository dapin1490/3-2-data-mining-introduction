import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

seed = 221214

data = pd.read_csv(r"weeks\week15\data\nofare2.csv", encoding="euc-kr", thousands = ',')
# print(data.columns)  # ['운행일자', '호선', '역', ' 총승차 ', ' 총하차 ']

data = data[['역', ' 총승차 ', ' 총하차 ']]
x = data[[' 총승차 ', ' 총하차 ']]
y = data['역']
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)

# 모델 훈련
logicr = LogisticRegression()
logicr.fit(x_tr, y_tr)
# logreg_acc = logicr.score(x_te, y_te)

# 예측
test_x = pd.DataFrame({' 총승차 ':[13000], ' 총하차 ':[12000]})
print(f"{logicr.predict(test_x)[0]}번째 역")