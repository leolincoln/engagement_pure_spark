import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__=='__main__':
    if len(sys.argv)<1:
        print 'error: python histbuilder name'
        file_name = 'heat3_19.csv'
    else:
        file_name = sys.argv[1]
    print 'using file_name',file_name
    #load data
    data = np.loadtxt(file_name)
    for i in range(len(data)):
        data[i][i]=0
    #flatten data
    data = np.ravel(data)
    #filter data
    #data = data[data<2]
    pd_data = pd.DataFrame(data)
    plt.figure()
    pd_data.plot.hist(bins=20)
    plt.savefig(file_name[:-4]+'.png')
