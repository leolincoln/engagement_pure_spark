#python heat3_temp.py 19 max_point_distance/ heat2_19.csv cluster_centers/
import sys
import platform
if platform.system()!='Linux':
    import matplotlib
    matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
from corr_matrix_subject import read_size,get_files,read_files_max,read_files_center
import numpy as np

if __name__=='__main__':
    if len(sys.argv)<3:
        print 'ERROR: sys.argv length'
        print 'Usage: python heat3.py subject# data_max_path u_path center_path'
        sys.exit(1)
    
    subject = sys.argv[1]
    print 'subject',subject

    data_max_path = sys.argv[2]
    print 'data_max_path',data_max_path

    u_path = sys.argv[3]
    print 'u_path',u_path
    
    center_path = sys.argv[4]
    print 'center_path',center_path
    top_list = read_size(subject = subject).index
    top_list = list(top_list)

    template = 'cluster_centers_subject'+str(subject)+'_.*csv'
    file_names = get_files(center_path,template)
    data = read_files_center(file_names)

    data_dict = {}
    for i in top_list:
        data_dict[i] = np.array(list(data.ix[i])).astype(float)
    del data

    #this is temp for u, its not actually u. Just csv from u.py
    u_temp = np.loadtxt(u_path)
    #this u is not real u. Its -u[a][b]-u[b][a] for each a and b in top_list[a] and top_list[b]
    u = np.zeros((len(top_list),len(top_list)),dtype=np.float)
    
    template_max = str(subject)+'_.*csv'

    file_names_max = get_files(data_max_path,template_max)
    data_max = read_files_max(file_names_max)


    for i in xrange(len(u_temp)):
        for j in xrange(len(u_temp[0])):
            c1 = top_list[i]
            c2 = top_list[j]
            if c1==c2:
                u[i][j]=0
                continue
            data_c1 = data_dict[c1]
            data_c2 = data_dict[c2]
            u_temp_data = u_temp[i][j]
            u[i][j]= u_temp_data-np.sqrt(sum((data_c1-data_c2)**2))+data_max.ix[top_list[i]]+data_max.ix[top_list[j]]
    np.savetxt('heat3_'+str(subject)+'.csv',u)
            

