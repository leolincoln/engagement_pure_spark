'''
to run this program, do something like this: ./runYarn.bash 'subject3' kmeans_subject.py 2>logs/kmeans_subject3.log where you have to specify subject number inside the program. 
I know this needs improving, but its working for now. I will fix it once we have all data. 
'''
#./runYarn.bash kmeans_subject.py 21 2>logs/kmeans_subject21.log
from pyspark import SparkContext
import os,itertools,time,math,sys
import numpy as np
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
import pickle


# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point, clusters):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

def error_by_center(point,clusters):
    num = clusters.predict(point)
    center = clusters.centers[num]
    return (num,sqrt(sum([x**2 for x in np.subtract(point,center)])))

def point_by_center(point,clusters):
    num = clusters.predict(point)
    center = clusters.centers[num]
    return (num,tuple(sqrt(sum([x**2 for x in np.subtract(point,center)]))),1)

def save_cluster_centers(centers,file_name):
    '''
    remove file_name file, create new file_name file, write item[0],item[1] for top_sizes in each line of file_name
    Args:
        top_sizes: list of list of feature items. 
        file_name: String of file_name output
    Returns:
        None
    '''
    os.system('rm -rf '+file_name)

    with open(file_name,'w') as f:
        for item in centers:
            #optimized, for improvement in speed
            f.write(','.join([str(item2) for item2 in item]))
            '''
            for item2 in item:
                f.write(str(item2))
                f.write(',')
            '''
            f.write('\n')

def save_cluster_sizes(top_sizes,file_name):
    '''
    remove file_name file, create new file_name file, write item[0],item[1] for top_sizes in each line of file_name
    Args:
        top_sizes: list of 2tuples. 
        file_name: String of file_name output
    Returns:
        None
    '''
    save_cluster_centers(top_sizes,file_name)
def xyz_feature(xyz_value):
    xyz_key = xyz_value[0]
    xyz_dict = xyz_value[1]
    features = [np.array(xyz_dict[key].split(',')).astype(float) for key in sorted(xyz_dict.keys())]
    merged = list(itertools.chain.from_iterable(features))
    return (xyz_key,merged) 

def xyz_subject_feature(xyz_value,subject ):
    '''
    Used for reducing
    Can be improved.
    Args:
        xyz_value: a 2tuple (xyz_key,xyz_dict)
        subject: the subject number to keep
    Returns:
        if subject in value, then (xyz_key, numpyArray of given subject)
        else (xyz_key, first value in xyz_dict.values())
    '''
    xyz_key = xyz_value[0]
    xyz_dict = xyz_value[1]
    if str(subject) in xyz_dict.keys():
        return (xyz_key,np.array(xyz_dict[str(subject)].split(',')).astype(float))
    else:
        return (xyz_key,np.array(xyz_dict.values()[0].split(',')).astype(float))

def value_pairs(line):
    '''
    get 'x,y,z' => values key=> value pairs
    Args:
        line is string of a line
        x;y;z;t0,t1,t2,...,tn
    '''
    values = line.split(';')
    #so now the first 3 are x,y and z.
    x=  values[0]
    y = values[1]
    z = values[2]
    subject = values[3]
    timeseries = values[4]
    return ((x,y,z),{subject:timeseries})

def xyz_group(xyz1,xyz2,subject):
    '''
    Update xyz1 and xyz2 if they have the same xyz_key
    Args:
        xyz1: a dictionary xyz_dict for key1
        xyz2: another dictionary xyz_dict for key1
    '''
    full = xyz1
    full.update(xyz2)
    return full

def print_rdd(rdd):
    '''
    Utility to print a rdd
    '''
    for x in rdd.collect():
        print x

def get_eud(values):
    pair1 = values[0]
    pair2 = values[1]
    xyz1 = pair1[0]
    xyz2 = pair2[0]
    v1 = [float(item) for item in pair1[1].split(',')]
    v2 = [float(item) for item in pair2[1].split(',')]
    result = sum([(v1[i]-v2[i])**2 for i in range(len(v1))])
    return ((xyz1,xyz2),math.sqrt(result))

def filter_0(line):
    array = [float(item) for item in line.split(';')[3]]
    return sum(array)!=0

def main(subject):
    time_now = time.time()
    sc = SparkContext()
    hdfsPrefix = 'hdfs://wolf.iems.northwestern.edu/user/huser54/'

    file_path1 = 'engagement/'
    file_path2 = 'engagementsample/'
    #TODO change the following parameters
    k0 = 500
    #k0 = 10
    sub_num = 10
    #sub_num = 1
    sub_k = 50
    #sub_k = 5
    #TODO change the file_path#,file_path1 is real data, file_path2 is sample data
    lines = sc.textFile(hdfsPrefix+file_path1)
    #map the values to xyz string -> dictionary of subjects with time series. 
    values = lines.map(value_pairs)
    print 'values obtained'
    #print values.first()
    #print 'value obtain time:',time.time()-time_now
    time_old = time.time()
    
    #group by key. Using reduce. Because groupby is not recommended in spark documentation
    groups = values.reduceByKey(lambda x,y:xyz_group(x,y,subject))
    print 'groups finished'
    #print groups.first()
    #print 'group obtain time:',time.time()-time_now
    time_now = time.time()

    #map the groups to xyz -> array, where array is 0-22 subject points. 
    feature_groups = groups.map(lambda x:xyz_subject_feature(x,subject))
    print 'feature group'
    #print feature_groups.first()
    #print 'feature obtain time:',time.time()-time_now
    time_now = time.time()

    parsedData = feature_groups.map(lambda x:x[1])
    parsedData.cache()
    print 'parsed data'

    #print parsedData.first()
    #print 'parsed data obtain time:',time.time()-time_now
    time_now = time.time()
    
    #now we have xyz -> group of features
    #and we are ready to cluster. 
    # Build the model (cluster the data)
    #document states:
    #classmethod train(rdd, k, maxIterations=100, runs=1, initializationMode='k-means||', seed=None, initializationSteps=5, epsilon=0.0001,initialModel=None)
    clusters = KMeans.train(parsedData, k0, maxIterations=100,runs=10, initializationMode="k-means||")
    print 'cluster obtain time:',time.time()-time_now
    time_now = time.time()

    WSSSE = parsedData.map(lambda point: error(point,clusters)).reduce(lambda x, y: x + y)
    os.system('rm -rf WSSE_subject'+str(subject)+'.dat')
    with open('WSSEs/WSSE_subject'+str(subject)+'.dat','w') as f:
        f.write(str(WSSSE))

    cluster_point_distance = parsedData.map(lambda point:error_by_center(point,clusters))
    cluster_point_distance.persist()

    max_point_distance = cluster_point_distance.reduceByKey(lambda x,y:max(x,y)).collect()
    save_cluster_sizes(max_point_distance,'max_point_distance/'+str(subject)+'_.csv')
    os.system('rm -rf max_point_distance/'+str(subject)+'_*.csv')

    sum_count_point_distance = cluster_point_distance.combineByKey(lambda value:(value,1.0),lambda x,value:(x[0]+value,x[1]+1), lambda x,y:(x[0]+y[0],x[1]+y[1]))

    average_point_distance = sum_count_point_distance.map(lambda (num,(value_sum,count)):(num,value_sum/count)).collect()
    save_cluster_sizes(average_point_distance,'average_point_distance/'+str(subject)+'_.csv')
    '''
    #Removed because duplicate to save_cluster_sizes
    os.system('rm -rf max_point_distance'+str(subject)+'.dat')
    with open('max_point_distance'+str(subject)+'.dat','w') as f:
        f.write(str(max_point_distance))
    '''
    time_now = time.time()

    #cluter centers after calculating kmeans clustering
    #clusterCenters = sc.parallelize(clusters.clusterCenters)

    #we dont need to clear hdfs system for now
    #print 'clearing hdfs system'
    #os.system('hdfs dfs -rm -r -f '+hdfsPrefix+'clusterCenters')
    cluster_ind = parsedData.map(lambda point:clusters.predict(point))
    cluster_ind.collect()
    cluster_sizes = cluster_ind.countByValue().items()
    #remove cluster size objects from cluster_sizes folder. 
    os.system('rm -rf cluster_sizes/cluster_sizes_subject'+str(subject)+'_*.csv')
    save_cluster_sizes(cluster_sizes,'cluster_sizes/cluster_sizes_subject'+str(subject)+'_.csv')
    #remove cluster_centers objects from cluster_centers folder. before we rewrite them
    os.system('rm -rf cluster_centers/cluster_centers_subject'+str(subject)+'_*.csv')
    centers_with_index = [[str(subject)+'_'+str(i)]+list(clusters.centers[i])for i in range(len(clusters.centers))]
    save_cluster_centers(centers_with_index,'cluster_centers/cluster_centers_subject'+str(subject)+'_.csv')
    del centers_with_index

    #get top clusters to split again
    top_clusters = [[item[0],int(item[1]/100)*2] for item in cluster_sizes if int(item[1]/100)>1]
    #cluster_point_distance.unpersist()

    #now we got the top 10 clusters. For each cluster, we will split 50 again. 
    for top_cluster in top_clusters:
        print 'processing',top_cluster
        top_data = parsedData.filter(lambda point:clusters.predict(point)==top_cluster[0])
        top_data.persist()
        #now temp_data has all filtered by top_cluster. 
        #Now we are going to cluster it. 
        #top_model = KMeans.train(top_data, sub_k, maxIterations=100,runs=10, initializationMode="k-means||")
        top_model = KMeans.train(top_data, top_cluster[1], maxIterations=100,runs=10, initializationMode="random")
        
        #top_data_point_distance = top_data.map(lambda point:error_by_center(point,top_model))
        #top_data_point_distance.persist()
        
        #top wsse
        top_wsse = top_data.map(lambda point: error(point,top_model)).reduce(lambda x, y: x + y)
        #group top data into different centers
        top_ind = top_data.map(lambda point:top_model.predict(point))
        top_ind.collect()
        #top_sizes are counts by subject. 
        top_sizes = top_ind.countByValue().items()
        #save cluster sizes
        save_cluster_sizes(top_sizes,'cluster_sizes/cluster_sizes_subject'+str(subject)+'_'+str(top_cluster[0])+'.csv')
        #save cluster centers
        centers_with_index = [[str(subject)+'_'+str(top_cluster[0])+'_'+str(i)]+list(top_model.centers[i]) for i in range(len(top_model.centers))]
        save_cluster_centers(centers_with_index,'cluster_centers/cluster_centers_subject'+str(subject)+'_'+str(top_cluster[0])+'.csv')
        #copied from above for max point to center distance. 
        cluster_point_distance = top_data.map(lambda point:error_by_center(point,top_model))
        #no need to persist because its different for each top cluster
        max_point_distance = cluster_point_distance.reduceByKey(lambda x,y:max(x,y)).collect()
        save_cluster_sizes(max_point_distance,'max_point_distance/'+str(subject)+'_'+str(top_cluster[0])+'.csv')
        print 'finished top cluster',top_cluster
        
        #begin top_cluster2, which is the second layer of top cluster        
        top_clusters2 = [[item[0],int(item[1]/100)*2] for item in top_sizes if int(item[1]/100)>1] 
        for top_cluster2 in top_clusters2:
            print 'processing',top_cluster
            top_data2 = top_data.filter(lambda point:top_model.predict(point)==top_cluster2[0])
            top_data2.persist()
            #now temp_data has all filtered by top_cluster. 
            #Now we are going to cluster it. 
            #top_model = KMeans.train(top_data2, sub_k, maxIterations=100,runs=10, initializationMode="k-means||")
            top_model2 = KMeans.train(top_data2, top_cluster2[1], maxIterations=100,runs=10, initializationMode="random")
            
            #top_data_point_distance = top_data.map(lambda point:error_by_center(point,top_model))
            #top_data_point_distance.persist()
            
            #top wsse
            top_wsse2 = top_data2.map(lambda point: error(point,top_model2)).reduce(lambda x, y: x + y)
            #group top data into different centers
            top_ind2 = top_data2.map(lambda point:top_model2.predict(point))
            top_ind2.collect()
            #top_sizes are counts by subject. 
            top_sizes2 = top_ind2.countByValue().items()
            #save cluster sizes
            save_cluster_sizes(top_sizes2,'cluster_sizes/cluster_sizes_subject'+str(subject)+'_'+str(top_cluster[0])+'_'+str(top_cluster2[0])+'.csv')
            #save cluster centers
            centers_with_index = [[str(subject)+'_'+str(top_cluster[0])+'_'+str(top_cluster2[0])+'_'+str(i)]+list(top_model.centers[i]) for i in range(len(top_model.centers))]
            save_cluster_centers(centers_with_index,'cluster_centers/cluster_centers_subject'+str(subject)+'_'+str(top_cluster[0])+'_'+str(top_cluster2[0])+'.csv')
            #copied from above for max point to center distance. 
            cluster_point_distance = top_data2.map(lambda point:error_by_center(point,top_model2))
            #no need to persist because its different for each top cluster
            max_point_distance = cluster_point_distance.reduceByKey(lambda x,y:max(x,y)).collect()
            save_cluster_sizes(max_point_distance,'max_point_distance/'+str(subject)+'_'+str(top_cluster[0])+'_'+str(top_cluster2[0])+'.csv')
            print 'finished top cluster 2',top_cluster
            top_data2.unpersist() 
        
        top_data.unpersist() 

    #save as text file to clusterCenters in hdfs
    print 'save cluster center',time.time()-time_now

    print 'wssse obtain time:',time.time()-time_old
    print("Within Set Sum of Squared Error = " + str(WSSSE))
if __name__=='__main__':
    if len(sys.argv)<2:
        print 'invalid parameters'
        print 'please append subject number'
        sys.exit(1)
    main(int(sys.argv[1]))
