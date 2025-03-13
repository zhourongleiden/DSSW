import numpy as np
import math

###############################################################################
# 3D cloud distance

def TDPCR_d(x,y,w1,w2):
    
    e=10**-30
    seg_1 = np.array(np.hsplit(x,3))
    seg_2 = np.array(np.hsplit(y,3))
    Ex1, En1 = seg_1[:,:,0], seg_1[:,:,1]
    Ex2, En2 = seg_2[:,:,0], seg_2[:,:,1]    

    En1[En1==0] = 1
    En2[En2==0] = 1

    Sup_s1 = Ex1+3*En1
    Inf_s1 = Ex1-3*En1
    Sup_s2 = Ex2+3*En2
    Inf_s2 = Ex2-3*En2    
    
    Ex1 = np.expand_dims(Ex1,2).repeat(len(x),axis=2)
    En1 = np.expand_dims(En1,2).repeat(len(x),axis=2)
    Ex2 = np.expand_dims(Ex2,1).repeat(len(y),axis=1)
    En2 = np.expand_dims(En2,1).repeat(len(y),axis=1)
    
    Sup_s1 = np.expand_dims(Sup_s1,2).repeat(len(x),axis=2)
    Inf_s1 = np.expand_dims(Inf_s1,2).repeat(len(x),axis=2)
    Sup_s2 = np.expand_dims(Sup_s2,1).repeat(len(y),axis=1)
    Inf_s2 = np.expand_dims(Inf_s2,1).repeat(len(y),axis=1)      
    
    OD = 2*(np.minimum(Sup_s1,Sup_s2)-np.maximum(Inf_s1,Inf_s2))/((Sup_s1-Inf_s1)+(Sup_s2-Inf_s2))
    OD[OD<0] = 0 
    
    flag = (En1 != En2)*2 + np.logical_and(En1 == En2, Ex1 != Ex2)*1 + np.logical_and(En1 == En2, Ex1 == Ex2)*-1   
    
    # flag = -1
    u_neg1 =  (flag == -1)*1
    OD[flag == -1] = 1
    #flag = 1
    x0 = (Ex1*En2+Ex2*En1)/(En1+En2)
    u_1 = (flag == 1)*np.exp(-(x0-Ex1)**2/(2*En1**2))
    #flag = 2
    a = (Ex2*En1-Ex1*En2)/(En1-En2+e)
    b = (Ex1*En2+Ex2*En1)/(En1+En2)
    x1 = np.minimum(a,b)
    x2 = np.maximum(a,b) 
    u1 = np.exp(-(x1-Ex1)**2/(2*En1**2)) 
    u2 = np.exp(-(x2-Ex2)**2/(2*En2**2))
    u_2 = (flag == 2)*np.maximum(u1,u2) 
    
    sim = OD*(u_neg1+u_1+u_2)    
    
    TOM_ZK = w1*sim[0]+w2*sim[1]+(1-w1-w2)*sim[2]
    dist_s1_s2 = 1/(TOM_ZK+e)-1
    
    return dist_s1_s2
###############################################################################


###############################################################################
# 3D cloud model
def cloud_model_algorithm_3D_matrix(time_series_source):
    
    axis = len(time_series_source.shape) - 1

    diff1 = np.diff(time_series_source, n=1)
    diff2 = np.diff(time_series_source, n=2)
    
    Ex1 = np.mean(time_series_source,axis=axis)
    En1 = np.mean(np.abs(time_series_source-np.expand_dims(Ex1,axis=axis)),axis=axis)*math.sqrt(math.pi/2)
    He1 = np.sqrt(np.abs(np.var(time_series_source,ddof=1,axis=axis)-En1**2))
    
    Ex2 = np.mean(diff1,axis=axis)
    En2 = np.mean(np.abs(diff1-np.expand_dims(Ex2,axis=axis)),axis=axis)*math.sqrt(math.pi/2)
    He2 = np.sqrt(np.abs(np.var(diff1,ddof=1,axis=axis)-En2**2))
    
    Ex3 = np.mean(diff2,axis=axis)
    En3 = np.mean(np.abs(diff2-np.expand_dims(Ex3,axis=axis)),axis=axis)*math.sqrt(math.pi/2)
    He3 = np.sqrt(np.abs(np.var(diff2,ddof=1,axis=axis)-En3**2))
    
    if axis == 1:
        return np.vstack((Ex1,En1,He1,Ex2,En2,He2,Ex3,En3,He3)).T 
    elif axis == 2:
        return np.stack((Ex1,En1,He1,Ex2,En2,He2,Ex3,En3,He3),axis=2)


