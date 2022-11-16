import matplotlib.pyplot as plt
import seaborn as sns  # tips, dots, anscombe, fmri 데이터셋 원래 출처
import pandas as pd
import numpy as np
import os

tips_file = 'data/tips.csv'
dots_file = 'data/dots.csv'
anscombe_file = 'data/anscombe.csv'
fm_file = 'data/fmri.csv'

if not os.path.exists("data/"):
	os.mkdir("data/")

if not os.path.exists("images/"):
	os.mkdir("images/")

if not os.path.exists(tips_file):
	tips = sns.load_dataset('tips')
	tips.to_csv(tips_file)

if not os.path.exists(dots_file):
	dots = sns.load_dataset('dots')
	dots.to_csv(dots_file)

if not os.path.exists(anscombe_file):
	dots = sns.load_dataset('anscombe')
	dots.to_csv(anscombe_file)

if not os.path.exists(fm_file):
	dots = sns.load_dataset('fmri')
	dots.to_csv(fm_file)

tips = pd.read_csv(tips_file)

# print(tips)
# print(type(tips))

sns.relplot(x='total_bill', y='tip', col='time', hue='smoker', style='smoker', size='size', data=tips)
# plt.show()
plt.savefig(r"images/ex01.png", facecolor='#dddddd', bbox_inches='tight')  # vscode는 r'weeks/week11/images/ex01.png'
plt.clf()
print("images/ex01.png")

dots = pd.read_csv(dots_file)

# kind='[]|line|scatter'
sns.relplot(x='time', y='firing_rate', col='align', hue='choice', size='coherence', style='choice',
            facet_kws=dict(sharex=False), kind='line', legend='full', data=dots)
# plt.show()
plt.savefig(r"images/ex02.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex02.png")

sns.relplot(x='time', y='firing_rate', col='align', hue='choice', size='coherence', style='choice',
            facet_kws=dict(sharex=False), legend='full', data=dots)
# plt.show()
plt.savefig(r"images/ex03.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex03.png")

sns.relplot(x='time', y='firing_rate', col='align', hue='choice', size='coherence', style='choice',
            facet_kws=dict(sharex=False), kind='scatter', legend='full', data=dots)
# plt.show()
plt.savefig(r"images/ex04.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex04.png")

"""
x = np.random.normal(size=100)
sns.distplot(x)  # UserWarning: `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
# plt.show()
plt.savefig(r"images/ex05.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex05.png")
"""

mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 30)
df = pd.DataFrame(data, columns=['x', 'y'])
sns.jointplot(x='x', y='y', data=df)
# plt.show()
plt.savefig(r"images/ex06.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex06.png")

sns.set(color_codes=True)
# tp = sns.load_dataset('tips')
tp = pd.read_csv(tips_file)
# ax = sns.regplot(x='total_bill', y='tip', data=tp)
sns.regplot(x='total_bill', y='tip', data=tp)
# plt.show()
plt.savefig(r"images/ex07.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex07.png")

np.random.seed(112)
mean, cov = [2, 3], [(1.5, 0.6), (0.6, 1)]
x, y = np.random.multivariate_normal(mean, cov, 30).T
# ax = sns.regplot(x=x, y=y, color='g')
sns.regplot(x=x, y=y, color='g')
# plt.show()
plt.savefig(r"images/ex08.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex08.png")

# ax = sns.regplot(x=x, y=y, ci=68)
sns.regplot(x=x, y=y, ci=68)
# plt.show()
plt.savefig(r"images/ex09.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex09.png")

# ax = sns.regplot(x='size', y='total_bill', data=tp, x_jitter=0.1)
sns.regplot(x='size', y='total_bill', data=tp, x_jitter=0.1)
# plt.show()
plt.savefig(r"images/ex10.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex10.png")

# ax = sns.regplot(x='size', y='total_bill', data=tp, x_estimator=np.mean)
sns.regplot(x='size', y='total_bill', data=tp, x_estimator=np.mean)
# plt.show()
plt.savefig(r"images/ex11.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex11.png")

tp['big_tip'] = (tp.tip / tp.total_bill) > 0.175
# ax = sns.regplot(x='total_bill', y='big_tip', data=tp, logistic=True, n_boot=500, y_jitter=.03)
sns.regplot(x='total_bill', y='big_tip', data=tp, logistic=True, n_boot=500, y_jitter=.03)
# plt.show()
plt.savefig(r"images/ex12.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex12.png")

# ans = sns.load_dataset('anscombe')
ans = pd.read_csv(anscombe_file)
# ax = sns.regplot(x='x', y='y', scatter_kws={'s': 200}, data=ans.loc[ans.dataset == 'II'], order=2, ci=None, truncate=True)
sns.regplot(x='x', y='y', scatter_kws={'s': 200}, data=ans.loc[ans.dataset == 'II'], order=2, ci=None, truncate=True)
# plt.show()
plt.savefig(r"images/ex13.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex13.png")

# ax = sns.regplot(x='x', y='y', scatter_kws={'s': 100}, data=ans.loc[ans.dataset == 'III'], robust=True, ci=None)
sns.regplot(x='x', y='y', scatter_kws={'s': 100}, data=ans.loc[ans.dataset == 'III'], robust=True, ci=None)
# plt.show()
plt.savefig(r"images/ex14.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex14.png")

sns.set(style='ticks', color_codes=True)
tips = pd.read_csv(tips_file)
# g = sns.FacetGrid(tips, col='time', row='smoker')
sns.FacetGrid(tips, col='time', row='smoker')
# plt.show()
plt.savefig(r"images/ex15.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex15.png")

g1 = sns.FacetGrid(tips, col='time', row='smoker')
g2 = g1.map(plt.hist, 'total_bill')
# plt.show()
plt.savefig(r"images/ex16.png", facecolor='#dddddd', bbox_inches='tight')
sns.set()
plt.clf()
print("images/ex16.png")

tips = pd.read_csv(tips_file)
# ax = sns.scatterplot(x='total_bill', y='tip', data=tips)
sns.scatterplot(x='total_bill', y='tip', data=tips)
# plt.show()
plt.savefig(r"images/ex17.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex17.png")

tips = pd.read_csv(tips_file)
# ax = sns.scatterplot(x='total_bill', y='tip', hue='time', style='time', size='size', data=tips)
sns.scatterplot(x='total_bill', y='tip', hue='time', style='time', size='size', data=tips)
# plt.show()
plt.savefig(r"images/ex18.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
sns.set()
print("images/ex18.png")

fm = pd.read_csv(fm_file)
# ax = sns.lineplot(x='timepoint', y='signal', data=fm)
sns.lineplot(x='timepoint', y='signal', data=fm)
plt.savefig(r"images/ex19.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex19.png")

# ax = sns.lineplot(x='timepoint', y='signal', hue='event', data=fm)
sns.lineplot(x='timepoint', y='signal', hue='event', data=fm)
plt.savefig(r"images/ex20.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex20.png")

# ax = sns.lineplot(x='timepoint', y='signal', hue='event', style='region', markers=True, dashes=False, data=fm)
sns.lineplot(x='timepoint', y='signal', hue='event', style='region', markers=True, dashes=False, data=fm)
# plt.show()
plt.savefig(r"images/ex21.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex21.png")

"""
# ax = sns.lineplot(x='timepoint', y='signal', hue='event', err_style='bars', ci=68, data=fm)
sns.lineplot(x='timepoint', y='signal', hue='event', err_style='bars', ci=68,
             data=fm)  # FutureWarning: The `ci` parameter is deprecated. Use `errorbar=('ci', 68)` for the same effect.
# plt.show()
plt.savefig(r"images/ex22.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex22.png")
"""

tip = pd.read_csv(tips_file)
sns.catplot(x='day', y='total_bill', data=tip)
# plt.show()
plt.savefig(r"images/ex23.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex23.png")

sns.catplot(x='day', y='total_bill', jitter=False, data=tip)
# plt.show()
plt.savefig(r"images/ex24.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex24.png")

sns.catplot(x='day', y='total_bill', kind='swarm', data=tip)
# plt.show()
plt.savefig(r"images/ex25.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex25.png")

sns.set(color_codes=True)
tp = pd.read_csv(tips_file)
sns.regplot(x='total_bill', y='tip', data=tp)
# plt.show()
plt.savefig(r"images/ex26.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex26.png")

sns.lmplot(x='total_bill', y='tip', data=tp)
# plt.show()
plt.savefig(r"images/ex27.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex27.png")

tp = pd.read_csv(tips_file)
sns.regplot(x='total_bill', y='tip', data=tp)
# plt.show()
plt.savefig(r"images/ex28.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex28.png")

sns.lmplot(x='total_bill', y='tip', data=tp, x_jitter=.05, x_estimator=np.mean)
# plt.show()
plt.savefig(r"images/ex29.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex29.png")

sns.lmplot(x='total_bill', y='tip', data=tp, hue='smoker', x_jitter=.05, x_estimator=np.mean, markers=['o', 'x'], palette='Set1', col='time', aspect=0.5)
# plt.show()
plt.savefig(r"images/ex30.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex30.png")

sns.jointplot(x='total_bill', y='tip', data=tp, kind='reg')
# plt.show()
plt.savefig(r"images/ex31.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex31.png")

sns.pairplot(tp, x_vars=['total_bill', 'size'], y_vars=['tip'], height=4, aspect=1.2, kind='reg')
# plt.show()
plt.savefig(r"images/ex32.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex32.png")

sns.set(style='ticks')
tp = sns.load_dataset('tips')
# g = sns.FacetGrid(tp, col='time')
sns.FacetGrid(tp, col='time')
# plt.show()
plt.savefig(r"images/ex33.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex33.png")


g = sns.FacetGrid(tp, col='time')
g.map(plt.hist, 'tip')
# plt.show()
plt.savefig(r"images/ex34.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex34.png")

g = sns.FacetGrid(tp, col='sex', hue='smoker')
g.map(plt.scatter, 'total_bill', 'tip', alpha=0.7)
g.add_legend()
# plt.show()
plt.savefig(r"images/ex35.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
print("images/ex35.png")
