import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(5, 4),   columns=['A', 'B', 'C', 'D'])
df.plot.box()
# plt.show()
plt.savefig(r"images/ex33.png", facecolor='#dddddd', bbox_inches='tight')
plt.clf()
