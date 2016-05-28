import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#load data
data = np.loadtxt('heat3_19.csv')
#flatten data
data = np.ravel(data)
pd_data = pd.DataFrame(data)
plt.figure()
pd_data.plot.hist(bins=20)
plt.savefig('test3.png')
