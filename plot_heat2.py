import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
r500 = np.loadtxt('heat2_19.csv')
sns.heatmap(r500, center=7)
plt.savefig('test.png')
