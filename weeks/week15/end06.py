import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc, font_manager

font_path = "c:/Windows/Fonts/HANDotum.ttf"  # 함초롬돋움
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

seed = 221214

data = pd.read_csv(r"weeks\week15\data\subway_1.csv", encoding="euc-kr", thousands = ',')
print(data.columns)  # ['사용일자', '노선명', '역명', '승차총승객수', '하차총승객수', '등록일자']

data = data[['노선명', '승차총승객수', '하차총승객수']]
x = data[['승차총승객수', '하차총승객수']]
y = data['노선명']
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)

# 모델 훈련
regr = linear_model.LinearRegression()
regr.fit(x_tr, y_tr)

sns.regplot(x='승차총승객수', y='하차총승객수', data=data)
plt.show()

# ??