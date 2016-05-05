#running command: python corr_matrix_subject.py 0 cluster_centers/ max_point_distance/
import platform
if platform.system()!='Linux':
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys,os,pandas,re
import pandas as pd
from os.path import join,getsize
from pandas import Series,DataFrame
#import matplotlib.pyplot as plt
from sklearn import metrics
import copy
import fnmatch
#find_match('src','*.csv')

def get_top(df,l=500):
    '''
    supposed the df was fed by read_subject_sizes where
    1. index should be subject_primarycluster_secondarycluster_number
    2. index should be of type str

    '''
    return list(df.sort_values(1,ascending=False).head(l).index)

def find_match(path,pattern):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

def read_size(subject):
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





def save_matrix_png(data,filename):
    plt.cla()
    #vmin = min(map(min,data))
    #vmax = max(map(max,data))
    #print vmin, vmax
    plt.matshow(data)
    #plt.clim(min(min(data)),max(max(data)))
    plt.colorbar()
    plt.savefig(filename)


def get_files(path,template):
    '''
        Args:
            path: the path to find files in
            template: the template to match names
        Returns: the list of files matching template in path
    will combile tempalte into pattern. 
    
    '''
    matches = []
    pattern = re.compile(template)
    for root, dirs,filenames in os.walk(path):
        for name in filter(lambda name:pattern.match(name),filenames):
            matches.append(join(root,name))
    return matches

def is_sub_cluster(file_path,template = None):
    return file_path.split('_')[-1]!='.csv'

def get_cluster_name(file_path):
    '''
    we say main# is the main cluster number,
    we also say sub# is the sub cluster number, 
    returns: 
    main#_sub#_0-49 if sub# is present. 
    main#_0=499 if no sub#
    '''
    if is_sub_cluster(file_path):
        main_num = main_number(file_path)
        sub_nums = sub_numbers(file_path)
        return [sub_nums+'_'+str(item) for item in range(50)]
    else:
        main_num = main_number(file_path)
        return [main_num +'_'+str(item) for item in range(500)]

def sub_numbers(file_path):
    '''
    e.g.cluster_centers/cluster_centers_subject0_40.csv
    should return 0_40
    getting the cluster number
    Args:
        file_path: the path of file
    Returns: 
        String of cluster number if sub cluster
        None if not
    '''
    #cluster_centers_subject0_40_2.csv
    file_path = os.path.basename(file_path)
    template = '([^0-9]*)(\d+_\d+_\d+)(.+)'
    pattern = re.compile(template)
    m = pattern.match(file_path)
    
    if m:
        cluster_number = m.groups()[-2]
        return cluster_number
    else:
        #cluster_centers_subject0_40.csv
        template = '([^0-9]*)(\d+_\d+)(.+)'
        pattern=re.compile(template)
        m = pattern.match(file_path)
        if m:
            return m.groups()[-2]
        else:
            return None
def main_number(file_path):
    '''
    e.g. cluster_centers/cluster_centers_subject0_40.csv
    should return 0
    Args:
        file_path: the path of file
    Returns:
        String of main cluster number if sub cluster
        None if not
    '''
    template = '([^0-9]*)(\d+)(_.+)'
    pattern = re.compile(template)
    m = pattern.match(file_path)
    if m:
        cluster_number = m.groups()[-2]
        return cluster_number
    else:
        return None

def read_file(file_path):
    '''
    read in a file and return a pandas dataframe. 
    Args:
        file_path: the string of file path
    '''
    data = pd.read_csv(file_path,header=None,index_col=0)
    return data

def read_files_center(file_paths):
    '''
    Args:
        file_paths: a list of string of file path. 
    Return:
        a pandas dataframe of all file_paths
    '''
    dfs = []
    for file_path in file_paths:
        data = read_file(file_path)
        #TODO: figure out why i did not put an extra ','
        #remove the last None because I put an extra ',' at the end of each line
        #data = data[data.columns[:-1]]
        dfs.append(data)
    return pd.concat(dfs)
def read_files_max(file_paths):
    '''
    Args:
        file_paths: a list of string of file path. 
    Return:
        a pandas dataframe of all file_paths
    '''
    #data frames
    dfs = []
    for file_path in file_paths:
        data = pd.read_csv(file_path,header=None,index_col=0)
        if is_sub_cluster(file_path):
            prefix = sub_numbers(file_path)+'_' 
        else:
            prefix = main_number(file_path)+'_'
        data.index = prefix+data.index.astype(str)
        dfs.append(data)
    return pd.concat(dfs)

if __name__=='__main__':
    if len(sys.argv)<4:
        print 'ERROR: no subject number supplied'
        print 'Usage: python corr_matrix_subject.py subject# cluster_center_path max_dis_path'
        print 'e.g. python corr_matrix_subject.py 0 cluster_centers/ max_point_distance/'
        sys.exit(1)
    subject = sys.argv[1]
    path = sys.argv[2]
   
   #print 'getting correlation matrix for subject',subject
    template = 'cluster_centers_subject'+str(subject)+'_.*csv'
    file_names = get_files(path,template)
    data = read_files_center(file_names)
    
    #obtain 1000*1000 cluster
    #top_list = get_top(read_size(subject = subject))

    #get the top sizes cluster name lists.
    top_list = read_size(subject = subject).index
    top_list = list(top_list)
    
    #tempalte for max distance
    template_max = str(subject)+'_.*csv'
    #get all files for max distance
    file_names_max = get_files(sys.argv[3],template_max)
    #read all max distance files indexed by cluster names
    data_max = read_files_max(file_names_max)
    
    #result for top 500 clusters
    r500 = np.zeros((len(top_list),len(top_list)),dtype=np.float)
    r500_2 = np.zeros((len(top_list),len(top_list)),dtype=np.float)
    count1_1 = 0
    count1_2 = 0
    count2_1 = 0
    count2_2 = 0
    count2 = 0
    count3 = 0
    max_center_distance = 0
    max_i = ''
    max_j = ''
    
    for i in xrange(len(r500)):
        for j in xrange(len(r500)):
            #first get the pairwise enclidean distance
            name_i = top_list[i]
            name_j = top_list[j]
            data_i = data.ix[name_i]
            data_j = data.ix[name_j]
            
            centroid_distance = float(np.sqrt(np.sum((data_i-data_j)**2)))
            
            if centroid_distance>max_center_distance:
                max_center_distance = centroid_distance
                max_i = name_i
                max_j = name_j
                max_plus = centroid_distance+float(data_max.ix[name_i]+data_max.ix[name_j])
                max_minus = centroid_distance-float(data_max.ix[name_i])-float(data_max.ix[name_j])
            r500[i][j]=centroid_distance+float(data_max.ix[name_i]+data_max.ix[name_j])
            r500_2[i][j] = centroid_distance-float(data_max.ix[name_i])-float(data_max.ix[name_j])
    
            #count of values on the right that are smaller than 1.34.
            if centroid_distance<=1.34:
                count1_1 += 1
            else:
                count1_2+=1
            if centroid_distance <=1.48:
                count2_1 +=1
            else:
                count2_2 +=1
            #the count of values on the left that are larger than 1.48.
    print subject,',',count1_1,',',count1_2,',',count2_1,',',count2_2
    print 'center_distance',max_center_distance,'max_i',max_i,float(data_max.ix[max_i]),'max_j',max_j,float(data_max.ix[max_j]),'max_plus',max_plus,'max_minus',max_minus
    np.savetxt('pluses'+str(subject)+'.csv',r500)
    np.savetxt('minuses'+str(subject)+'.csv',r500_2)
    save_matrix_png(r500,'pluses'+str(subject)+'.png')
    save_matrix_png(r500_2,'minuses'+str(subject)+'.png')
    
    '''

            if r500_2[i][j]>0.000001:
                count1 += 1 
            #count of values on the right that are smaller than 1.34.
            if r500[i][j]<1.34:
                count2 += 1
            #the count of values on the left that are larger than 1.48.
            if r500_2[i][j]>1.48:
                count3 += 1



    result = metrics.pairwise.pairwise_distances(data)
    result2 = copy.copy(result)
    count1 = 0
    count2 = 0
    count3 = 0
    cluster_names = list(cluster_names)
    
    #show results for top 500 clusters

    #print 'original cluster names',len(cluster_names)
    #print 'top list length',len(top_list)
    #print 'cluster names - top list',len(set(cluster_names)-set(top_list))
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j]+=data_max.ix[cluster_names[i]]
            result2[i][j]-=data_max.ix[cluster_names[i]]
            result[i][j]+=data_max.ix[cluster_names[j]]
            result2[i][j]-=data_max.ix[cluster_names[j]]
            if cluster_names[i] in top_list and cluster_names[j] in top_list:
                #print 'both i and j are in top list',i,j
                #count of values on the left that are larger than 0.000001
                if result2[i][j]>0.000001:
                    count1 +=1 
                #count of values on the right that are smaller than 1.34.
                if result[i][j]<1.34:
                    count2 +=1
                #the count of values on the left that are larger than 1.48.
                if result2[i][j]>1.48:
                    count3+=1
            else:
                result[i][j] = 0
                result2[i][j]=2
    print subject,',',count1,',',count2,',',count3
    plt.matshow(result)
    plt.colorbar()
    plt.savefig('pluses'+str(subject)+'.png')

    plt.cla()
    plt.matshow(result2)
    plt.colorbar()
    plt.savefig('minuses'+str(subject)+'.png')
    '''
