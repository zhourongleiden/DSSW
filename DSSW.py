import pandas as pd
import numpy as np
import math
import time
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from cloud_model import TDPCR_d, cloud_model_algorithm_3D_matrix
from tslearn.metrics import dtw_path_from_metric, dtw
from args import args

def DTW_parellel(iteration):
    if arguments.alg == "DSSW":
        iteration_i, iteration_j, w1, w2 = iteration
        s1 = seg_representation_all_TEST[iteration_i]
        s2 = seg_representation_all_TRAIN[iteration_j]       
        dist_s1_s2 = TDPCR_d(s1,s2,w1,w2)  
        distance_instance = dtw_path_from_metric(dist_s1_s2, metric="precomputed")[1]       
    else:
        iteration_i, iteration_j = iteration
        s1 = data_original_TEST[iteration_i]
        s2 = data_original_TRAIN[iteration_j]    
        distance_instance = dtw(s1,s2)
    i_result = [iteration_i, iteration_j, distance_instance]
    
    return i_result    


def DTW_1NN(flag):
    print("##############Start DTW_1NN###############")
    start = time.perf_counter()
    
    # DTW_parellel
    dist_dataframe = []
    iteration = []
    for i in range(0,len(data_original_TEST if flag == "out" else data_original_TRAIN)):
        for j in range(0,len(data_original_TRAIN)):
            iteration.append((i,j))   
    with Pool(int(multiprocessing.cpu_count()/3)) as p:
        max_ = len(iteration)
        with tqdm(total = max_) as pbar:
            for item in p.imap_unordered(DTW_parellel, iteration):
                dist_dataframe.append(item)
                pbar.update()
    
    # reconstruction             
    dist_instance = np.zeros((len(data_original_TEST if flag == "out" else data_original_TRAIN),len(data_original_TRAIN)))
    for i in range(0,len(dist_dataframe)):
        dist_instance[dist_dataframe[i][0],dist_dataframe[i][1]] = dist_dataframe[i][2]      
          
    # KNN
    if flag == "in":
        dist_instance += np.eye(len(data_original_TRAIN))*1e99 # 排除跟自己的距离
    instance_predict = np.argmin(dist_instance, axis = 1).tolist()
    label_predict = [label_TRAIN[instance_predict[i]] for i in range(0,len(instance_predict))]
    
    # calculate classification accuracy
    accuracy = 0
    for i in range(0,len(instance_predict)):
        label = label_TEST if flag == "out" else label_TRAIN
        if label_predict[i] == label[i]:
            accuracy = accuracy + 1
    accuracy = accuracy / len(instance_predict) 
    
    end = time.perf_counter()
    print("Dataname: {}, Execution time: ".format(arguments.dataname), format(end-start,'.3f')) 
    
    return 1-accuracy


def DSSW_1NN():
    print("##############Start DSSW_1NN###############")
    w1 = arguments.w1
    w2 = arguments.w2
    noc = arguments.noc
    if (w1 > 0.5 and w2>(1-w1)):
        w1 = 1 - w1
        w2 = 1 - w2
    print("noc={}, w1={}, w2={}, w3={}".format(noc,round(w1,3),round(w2,3),round(1-w1-w2,3)))
    
    # segmentation
    start = time.perf_counter()
    
    global seg_representation_all_TRAIN
    seg_representation_all_TRAIN = []  
    seg_TRAIN_length = math.floor(data_original_TRAIN.shape[1]/noc)
    seg_TRAIN_index = list(filter(lambda x: (x % seg_TRAIN_length == 0), np.arange(data_original_TRAIN.shape[1])))
    if noc > 1 :
        del seg_TRAIN_index[0]
        if data_original_TRAIN.shape[1] % seg_TRAIN_length != 0:
            del seg_TRAIN_index[-1]
        seg_TRAIN = np.split(data_original_TRAIN, seg_TRAIN_index, axis=1)
        last_seg_TRAIN = seg_TRAIN[-1]
        del seg_TRAIN[-1]
    else:
        seg_TRAIN = [data_original_TRAIN]
    seg_TRAIN = np.stack(seg_TRAIN)
    seg_representation_all_TRAIN.append(cloud_model_algorithm_3D_matrix(seg_TRAIN))
    if noc > 1:
        seg_representation_all_TRAIN.append(np.expand_dims(cloud_model_algorithm_3D_matrix(last_seg_TRAIN), axis=0))
    seg_representation_all_TRAIN = np.vstack(seg_representation_all_TRAIN)
    seg_representation_all_TRAIN = np.swapaxes(seg_representation_all_TRAIN, 0, 1)
    
    global seg_representation_all_TEST
    seg_representation_all_TEST = []
    seg_TEST_length = math.floor(data_original_TEST.shape[1]/noc)
    seg_TEST_index = list(filter(lambda x: (x % seg_TEST_length == 0), np.arange(data_original_TEST.shape[1])))
    if noc > 1 :
        del seg_TEST_index[0] #delete "0"
        if data_original_TEST.shape[1] % seg_TEST_length != 0 : #delete the last index, e.g. 721 % 360 != 0 so the last segment index is 720, we only want [360]
            del seg_TEST_index[-1]
        seg_TEST = np.split(data_original_TEST, seg_TEST_index, axis=1)
        last_seg_TEST = seg_TEST[-1]
        del seg_TEST[-1]
    else:
        seg_TEST = [data_original_TEST]
    seg_TEST = np.stack(seg_TEST)
    seg_representation_all_TEST.append(cloud_model_algorithm_3D_matrix(seg_TEST))
    if noc > 1:
        seg_representation_all_TEST.append(np.expand_dims(cloud_model_algorithm_3D_matrix(last_seg_TEST), axis=0))
    seg_representation_all_TEST = np.vstack(seg_representation_all_TEST)
    seg_representation_all_TEST = np.swapaxes(seg_representation_all_TEST, 0, 1)
    
    end = time.perf_counter()
    print("Dataname: {}, Execution time: ".format(arguments.dataname), format(end-start,'.3f'))    
    
    # DTW_parellel
    dist_dataframe = []
    iteration = []
    for i in range(0,len(data_original_TEST)):
        for j in range(0,len(data_original_TRAIN)):
            iteration.append((i,j,w1,w2))   
    with Pool(int(multiprocessing.cpu_count()/3)) as p:
        max_ = len(iteration)
        with tqdm(total = max_) as pbar:
            for item in p.imap_unordered(DTW_parellel, iteration):
                dist_dataframe.append(item)
                pbar.update()
   
    # Reconstruction             
    dist_instance = np.zeros((len(data_original_TEST),len(data_original_TRAIN)))
    for i in range(0,len(dist_dataframe)):
        dist_instance[dist_dataframe[i][0],dist_dataframe[i][1]] = dist_dataframe[i][2]      
      
    # KNN
    instance_predict = np.argmin(dist_instance, axis = 1).tolist()
    label_predict = [label_TRAIN[instance_predict[i]] for i in range(0,len(instance_predict))]  
    
    # Calculate classification accuracy
    accuracy = 0
    for i in range(0,len(instance_predict)):
        label = label_TEST
        if label_predict[i] == label[i]:
            accuracy = accuracy + 1
    accuracy = accuracy / len(instance_predict) 
    end = time.perf_counter()
    print("Dataname: {}, Execution time: ".format(arguments.dataname), format(end-start,'.3f'))
        
    return 1-accuracy

#############################################################################################
if __name__ == '__main__':    
    # Load args
    arguments = args()
    print("start testing on {}".format(arguments.dataname))
    # Load data
    csv_data_TRAIN_all = np.array(pd.read_csv(arguments.csv_data_TRAIN_path, header=None))
    csv_data_TEST = np.array(pd.read_csv(arguments.csv_data_TEST_path, header=None))    
    # Create label and data
    label_TRAIN = csv_data_TRAIN_all[:,0]
    data_original_TRAIN = np.delete(csv_data_TRAIN_all, obj=0, axis=1)
    label_TEST = csv_data_TEST[:,0]  
    data_original_TEST = np.delete(csv_data_TEST, obj=0, axis=1)  
    # Initialise
    seg_representation_all_TRAIN = []
    seg_representation_all_TEST= []
    # Test model
    if arguments.alg == "DSSW":      
        error = DSSW_1NN()
        print("dataname={}, classification_error={}".format(arguments.dataname, round(error,3)))                      
    else:
        error = DTW_1NN(flag="out")    
        print("dataname={}, classification_error={}".format(arguments.dataname, round(error,3)))  
       