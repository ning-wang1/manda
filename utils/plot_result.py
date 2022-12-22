import pandas as pd
import os
import seaborn as sns

from matplotlib import pyplot as plt

csv_read = pd.read_csv('result/1sum.csv')
print(csv_read['adversarial'])
adv = pd.Series(csv_read['adversarial'])
original = pd.Series(csv_read['original'])
plt.figure()
# adv.plot(kind='kde')
# original.plot(kind='kde')
# sns.kdeplot(adv, shade = True)
# sns.distplot(csv_read['adversarial'], hist=False)
uncertainty_df = pd.DataFrame(csv_read, columns=['adversarial', 'original'])
ax = uncertainty_df.plot.box(grid=True)
plt.ylabel('u(x)')
plt.show()


# import pandas as pd
# import numpy as np
# import os
# import seaborn as sns

# from matplotlib import pyplot as plt

# csv_read = pd.read_csv('result/1sum.csv')
# print(csv_read['adversarial'])
# adv = np.array(csv_read['adversarial'])
# original = np.array(csv_read['original'])

# # weights = np.ones_like(adv)/float(len(adv))
# # plt.hist(adv, bins=3, weights=weights)
# #
# #
# # weights = np.ones_like(original)/float(len(original))
# # plt.hist(original, bins=3, weights=weights)

# # adv.plot(kind='kde')
# # original.plot(kind='kde')

# sns.distplot(csv_read['adversarial'])
# sns.distplot(csv_read['original'])
# # plt.hist(adv, density=True)
# # plt.hist(original, density=True)
# # uncertainty_df = pd.DataFrame(csv_read, columns=['adversarial', 'original'])
# # ax = uncertainty_df.plot.box(grid=True)
# plt.show()
