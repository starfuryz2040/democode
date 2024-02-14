# -*- coding: utf-8 -*-

from Precode2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.load('AllSamples.npy')

k1,i_point1,k2,i_point2 = initial_S2('0000')


def kmeans(k, initialpoints):
    
    topdist = 0.0
    ipt1=initialpoints.reshape(1,2)
    
    centroids=[]
    lasttop=np.empty((0,2))
    #centroids=centroids.reshape(1,2)
    centroids=ipt1
    
    centroiddata = data.copy()
    
    #find centroids
    while(len(centroids)<k):
        
        topdist = 0.0
        lasttop=np.empty((0,2))
        
        for pt in data:
            dist = 0.0
            point=pt.reshape(1,2)
            #print(pt)
            if point not in centroids:
                for cp in centroids:
                    dist = dist + np.linalg.norm(point-cp)
                dist=dist/len(centroids)
                if dist > topdist:
                    topdist = dist
                    lasttop = point
            #print(topdist)
    
        centroids=np.concatenate((centroids,lasttop), axis=0)
    
    firstcentroids = centroids
    
    data2=np.column_stack((data, np.zeros(np.shape(data)[0])))
    data2=np.column_stack((data2, np.zeros(np.shape(data2)[0])))
    
    pos=0
    
    #assign pts to centroids
    for pts in data2:
        
        dist = 0.0
        leastdist = float('inf')
        lastcentroid=np.empty((0,2))
        
        points=pts[:2]
        for cp in centroids:
            cp=cp.reshape(1,2)
            dist = np.linalg.norm(points-cp)
            if dist < leastdist:
                lastcentroid = cp
                leastdist = dist
        data2[pos][2]=lastcentroid[0][0]
        data2[pos][3]=lastcentroid[0][1]
        pos=pos+1
    
    lastsse = float('inf')
    currentsse = 0.0
    
    dataset = pd.DataFrame({'ptx': data2[:, 0], 'pty': data2[:, 1],
                            'ctx': data2[:, 2], 'cty': data2[:, 3] })
    
    data3=data2.copy()
    
    
    equal_arrays = False
    
    while(equal_arrays == False):
        pos = 0
        
        getNewCentroids = dataset.groupby(['ctx', 'cty']).agg(
        {
             'ptx':'mean',    # Sum duration per group
             'pty':'mean'   # get the count of networks
        }
        )
        
        centx = getNewCentroids['ptx'].to_numpy()
        centy = getNewCentroids['pty'].to_numpy()
        newcentroids = np.column_stack((centx, centy))
        
        olddf = data3.copy()
        
        for pts in data3:
        
            dist = 0.0
            leastdist = float('inf')
            lastcentroid=np.empty((0,2))
            #print("pts start")
            #print(pts)
            
            points=pts[:2]
            for cp in newcentroids:
                cp=cp.reshape(1,2)
                dist = np.linalg.norm(points-cp)
                #print("{} to {}".format(dist, cp ))
                if dist < leastdist:
                    lastcentroid = cp
                    leastdist = dist
            #print(lastcentroid)
            data3[pos][2]=lastcentroid[0][0]
            data3[pos][3]=lastcentroid[0][1]
            pos=pos+1
        
        dataset = pd.DataFrame({'ptx': data3[:, 0], 'pty': data3[:, 1],
                            'ctx': data3[:, 2], 'cty': data3[:, 3] })
        
        comparison = olddf == data3
        equal_arrays = comparison.all()
    
    total = 0
    sse = 0
     
    for errs in data3:
        pterr=errs.reshape(1,4)
        a = pterr[:,[0,1]]
        b = pterr[:,[2,3]]
        dist = np.linalg.norm(a-b)
        dist = dist**2
        sse = sse + dist
    
    
    print(newcentroids)
    print(sse)
    

    
    plt.scatter(x=dataset['ptx'], y=dataset['pty'], c=dataset['ctx'], alpha=0.5,
                cmap="viridis")
    plt.scatter(x=dataset['ctx'], y=dataset['cty'], c='b', alpha=0.5)
            

    plt.show()




print("project initialization")
print("k1:= {}".format(k1))
print("k1 intial centroids")
print(i_point1)
print("k2:= {}".format(k2))
print("k2 intial centroids")
print(i_point2)

print("*******starting keans algo********")
print("k1 centroids and sse")
kmeans(k1, i_point1)
print("*******round2 keans algo********")
print("k2 centroids and sse")
kmeans(k2, i_point2)
    
