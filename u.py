import platform
if platform.system()!='Linux':
    import matplotlib
    matplotlib.use('TkAgg')
import numpy as np
import sys,os,re
import pandas as pd
import matplotlib.pyplot as plt
from corr_matrix_subject import read_files_center,get_files,read_size

def cluster_predict(data_dict,data_in):
    '''
    data_dict has index->list
    data_in has a list of numbers. 
    Returns:
        index of which cluster data_in belongs to
    '''
    
    #find the smalleast distance from data_dict to data_in
    result = test.keys()[0]
    data_in = np.array(data_in)
    min_dis = sum((data_in-test[result])**2)

    for item in data_dict.keys():
        temp = sum((data_in-test[i])**2)
        if temp < min_dis:
            min_dis = temp
            result = item
    return result
    
def value_pairs(line):
    '''
    Copied from kmeans.py
    '''
    values = line.split(';')
    #so now the first 3 are x,y and z.
    x=  values[0]
    y = values[1]
    z = values[2]
    subject = values[3]
    timeseries = values[4]
    return ((x,y,z),{subject:timeseries})

if __name__=='__main__':
    if len(sys.argv)<1:
        print 'ERROR: sys.argv length'
        print 'Usage: python u.py subject # data_path cluster_center_path '
        sys.exit(1)
    subject = sys.argv[1]
    data_path = sys.argv[2]

    #between any two clusters. Therefore, we need to get all clusters first. 
    top_list = read_size(subject = subject).index
    top_list = list(top_list)
    
    #read centroids from files. 
    template = 'cluster_centers_subject'+str(subject)+'_.*csv'
    file_names = get_files(data_path,template)
    data = read_files_center(file_names)
    data_dict = {}
    for i in top_list:
        data_dict[i] = np.array(list(data.ix[i]))
    del data
    #now data_dict has index -> list where index is centroid name, list is a list of data points

    #read data from hdfs
    sc = SparkContext()
    file_path1 = 'engagement/'
    file_path2 = 'engagementsample/'
    hdfsPrefix = 'hdfs://wolf.iems.northwestern.edu/user/huser54/'

    #reading data from hdfs
    lines = sc.textFile(hdfsPrefix+file_path1)
    
    #get subject data from hdfs
    subject_data = lines.filter(lambda x:str(x.split(';')[3])==str(subject))
    
    
    values = lines.map(value_pairs)

