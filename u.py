import platform
if platform.system()!='Linux':
    import matplotlib
    matplotlib.use('TkAgg')
from pyspark import SparkContext
import numpy as np
import sys,os,re
import pandas as pd
import matplotlib.pyplot as plt
from corr_matrix_subject import read_files_center,get_files,read_size,save_matrix_png

'''
I am lazy as not wanting to create pakage for my previous programs. 
'''

def cluster_predict(data_dict,data_in):
    '''
    data_dict has index->list
    data_in has a list of numbers. 
    Returns:
        index of which cluster data_in belongs to
    '''
    
    #find the smalleast distance from data_dict to data_in
    result = data_dict.keys()[0]
    data_in = np.array(data_in).astype(float)
    print 'in cluster_predict'
    print '*'*20
    print len(data_in)
    print len(data_dict[result])
    print '*'*20
    min_dis = sum((data_in-data_dict[result])**2)

    for item in data_dict.keys():
        temp = sum((data_in-data_dict[item])**2)
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
def group_list(x,y):
    x.extend(y)
    return x
def sort_list(x,data_dict):
    '''
    Params:
        x: the rdd element, x[0] is the centroid_name, x[1] is the list of real data 
        data_dict: centroid_name-> actual_centroid_data
    '''
    center = data_dict[x[0]]
    sorted_list = sorted(x[1],key=lambda x:sum((x-center)**2),reverse=True)
    return (x[0],sorted_list)

def minusminus(x,data_dict,sorted_values):
    '''
    x: 2 centroid names
    data_dict: the centroid_name->centroid_data dict
    sorted_values: the centroid_name->sorted list of data by distance

    '''
    c1 = x[0][0][0]
    c2 = x[0][1][0]
    data_c1 = data_dict[c1]
    data_c2 = data_dict[c2]
    #U(ab) = max(||X-Ca||), (Ca-Cb)*(x-Ca)>0
    key = (c1,c2)
    x_list = x[0][0][1] 
    for data in x_list:
        if sum((data_c1-data_c2)*(data-data_c1))>0:
            return (key,np.sqrt(sum((data-data_c1)**2)))
    return (key,-5)
def calculate_heatmap(x,data_dict,u):
    '''
    
    '''
    c1 = x[0][0]
    c2 = x[0][1]
    data_c1 = data_dict[c1]
    data_c2 = data_dict[c2]
    #centroid distance (a,b) - u(ab) - u(ba)
    key = (c1,c2)
    return (key,np.sqrt(sum((data_c1-data_c2)**2))-u[(c1,c2)]-u[(c2,c1)])
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
    #data_dict is the centroids dictionary
    data_dict = {}
    for i in top_list:
        data_dict[i] = np.array(list(data.ix[i])).astype(float)
    del data
    #now data_dict has index -> np float array where index is centroid name, list is a list of data points

    #read data from hdfs
    sc = SparkContext()
    file_path1 = 'engagement/'
    file_path2 = 'engagementsample/'
    hdfsPrefix = 'hdfs://wolf.iems.northwestern.edu/user/huser54/'

    #reading data from hdfs
    lines = sc.textFile(hdfsPrefix+file_path1)
    
    #get subject data from hdfs
    subject_data = lines.filter(lambda x:str(x.split(';')[3])==str(subject))
    test = subject_data.take(1)
    print test,len(test[0].split(';')[4].split(','))
    print len(data_dict.values()[0])
    subject_value = subject_data.map(lambda x:np.array(x.split(';')[4].split(',')).astype(float))

    #map data to different centers
    values = subject_value.map(lambda x:(cluster_predict(data_dict,x),[x]))
    grouped_values = values.reduceByKey(lambda x,y:group_list(x,y))
    sorted_values = grouped_values.map(lambda x:sort_list(x,data_dict))
    cross_centroids = sorted_values.cartesian(sorted_values) 
    u = cross_centroids.map(lambda x:minusminus(x,data_dict))
    u_dict = {}
    for element in u.collect():
        u_dict[element[0]] = element[1]
    #centroid distance (a,b) - u(ab) - u(ba)
    heatmap = cross_centroids.map(lambda x:calculate_heatmap(x,data_dict,u_dict))
            
    r500 = np.zeros((len(top_list),len(top_list)),dtype=np.float)
    for element in heatmap.collect():
        c1 = element[0][0]
        c2 = element[0][1]
        i = top_list.index(c1)
        j = top_list.index(c2)
        r500[i][j] = element[1]

    save_matrix_png(r500,str(subject)+'.png')




