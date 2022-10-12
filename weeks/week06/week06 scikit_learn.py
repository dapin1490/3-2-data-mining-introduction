from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

if not os.path.exists("data/"):
	os.mkdir("data/")

url = 'http://bit.ly/kaggletrain'
train = pd.read_csv(url)

feature_cols = ['Pclass', 'Parch']
X = train.loc[:, feature_cols]
y = train.Survived

print(X.head, end="\n\n")
print(y.head, end="\n\n")

logreg = LogisticRegression()
logreg.fit(X, y)
url_test = 'http://bit.ly/kaggletest'
test = pd.read_csv(url_test)
X_new = test.loc[:, feature_cols]
new_pred_class = logreg.predict(X_new)

print(new_pred_class[0:5], end="\n\n")

kaggle_data = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': new_pred_class}).set_index('PassengerId')
kaggle_data.to_csv('data/sub.csv')

train.to_pickle('data/train.pkl')
pd.read_pickle('data/train.pkl')
