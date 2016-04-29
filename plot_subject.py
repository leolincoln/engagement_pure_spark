import platform
if platform.system()!='Linux':
    import matplotlib
    matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import sys
import pandas as pd
from glob import glob
subject = 7
import fnmatch
import os
#find_match('src','*.csv')
def find_match(path,pattern):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

def read_subject_sizes2(subject = subject):
    #TODO FIXED
    #cheange the subject so that it wont check for subject 11 for subject 1
    file_names = find_match('cluster_sizes','cluster_sizes_subject'+str(subject)+'_*.csv')
    frames = []
    #centroids store the original centroids with no sub cluster, e.g. the main centroids
    centroids = set([])
    for f in file_names:
        print 'processing',f
        #because the upper clusters are somethingsubjectnumber_.csv
        #we can see only .csv at the last split
        if f.split('_')[-1]=='.csv':
            new_frame = pd.read_csv(f,index_col=0,header=None)
            prefix = str(subject)+'_'
            new_frame.index = prefix+new_frame.index.astype(str)
            frames.append(new_frame)
        else:
            cluster_number_prefix = str(subject)+'_'+f.split('_')[-1].split('.')[0]+'_'
            centroids.add(cluster_number_prefix[:-1]) 
            new_frame = pd.read_csv(f,index_col=0,header=None)
            new_frame.index = cluster_number_prefix+new_frame.index.astype(str)        
            frames.append(new_frame)      
            print len(frames)
    allframe =  pd.concat(frames)
    #return allframe
    return allframe.ix[allframe.index.astype(str).isin(centroids)]
def read_subject_sizes(subject = subject):
    #TODO FIXED
    #cheange the subject so that it wont check for subject 11 for subject 1
    file_names = find_match('cluster_sizes','cluster_sizes_subject'+str(subject)+'_*.csv')
    frames = []
    #centroids store the original centroids with no sub cluster, e.g. the main centroids
    centroids = set([])
    for f in file_names:
        print 'processing',f
        #cluster_sizes_subject21_466_72.csv
        if len(os.path.basename(f).split('_'))==5:
            cluster_number_prefix = str(subject)+'_'+f.split('_')[-2]+'_'+f.split('_')[-1].split('.')[0]+'_'
            centroids.add(cluster_number_prefix[:-1]) 
            new_frame = pd.read_csv(f,index_col=0,header=None)
            new_frame.index = cluster_number_prefix+new_frame.index.astype(str)        
            frames.append(new_frame)      
            print len(frames)

        #because the upper clusters are somethingsubjectnumber_.csv
        #we can see only .csv at the last split
        elif f.split('_')[-1]=='.csv':
            new_frame = pd.read_csv(f,index_col=0,header=None)
            prefix = str(subject)+'_'
            new_frame.index = prefix+new_frame.index.astype(str)
            frames.append(new_frame)
        else:
            cluster_number_prefix = str(subject)+'_'+f.split('_')[-1].split('.')[0]+'_'
            centroids.add(cluster_number_prefix[:-1]) 
            new_frame = pd.read_csv(f,index_col=0,header=None)
            new_frame.index = cluster_number_prefix+new_frame.index.astype(str)        
            frames.append(new_frame)      
            print len(frames)
    allframe =  pd.concat(frames)
    #return allframe
    return allframe.ix[~allframe.index.astype(str).isin(centroids)]
def get_top_500_sizes(df,l=500):
    '''
    supposed the df was fed by read_subject_sizes where
    1. index should be subject_primarycluster_secondarycluster_number
    2. index should be of type str

    '''
    return list(df.sort_values(1,ascending=False).head(l).index)
if __name__=='__main__':
    print 'subject number:', sys.argv[1]
    data = read_subject_sizes(subject = sys.argv[1])
    #print get_top_500_sizes(data)
