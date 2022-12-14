import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

seed = 221214
k = 8

data = pd.read_csv(r"weeks\week15\data\subway_2.csv", encoding="euc-kr")
# print(data.columns)  # ['사용일자', '노선명', '역명', '승차총승객수', '하차총승객수', '등록일자']

data_5 = data[['노선명', '승차총승객수', '하차총승객수']]
x = data_5[['승차총승객수', '하차총승객수']]
y = data['노선명']
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)

# 각 모델 훈련
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_tr, y_tr)
knn_acc = knn.score(x_te, y_te)

logicr = LogisticRegression()
logicr.fit(x_tr, y_tr)
logreg_acc = logicr.score(x_te, y_te)

clf = DecisionTreeClassifier()
clf.fit(x_tr, y_tr)
clf_acc = clf.score(x_te, y_te)

# 예측
test_x = pd.DataFrame({'승차총승객수':[30000], '하차총승객수':[30000]})
print(f"""
--- predict ---
KNeighborsClassifier : {knn.predict(test_x)[0]}
LogisticRegression : {logicr.predict(test_x)[0]}
DecisionTreeClassifier : {clf.predict(test_x)[0]}
""")


best = ""
if knn_acc > logreg_acc and knn_acc > clf_acc:
    best = "KNeighborsClassifier"
elif logreg_acc > knn_acc and logreg_acc > clf_acc:
    best = "LogisticRegression"
else:
    best = "DecisionTreeClassifier"

print(f"""
--- accuracy ---
KNeighborsClassifier : {knn_acc}
LogisticRegression : {logreg_acc}
DecisionTreeClassifier : {clf_acc}

best : {best}
""")