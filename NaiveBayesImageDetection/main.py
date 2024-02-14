# -*- coding: utf-8 -*-
"""
"""

import numpy
import scipy.io
import math


Numpyfile0 = scipy.io.loadmat('digit0_stu_train0000.mat')
Numpyfile1 = scipy.io.loadmat('digit1_stu_train0000.mat')
Numpyfile2 = scipy.io.loadmat('digit0_testset.mat')
Numpyfile3 = scipy.io.loadmat('digit1_testset.mat')


train0 = Numpyfile0.get('target_img')
train1 = Numpyfile1.get('target_img')
test0 = Numpyfile2.get('target_img')
test1 = Numpyfile3.get('target_img')



print([len(train0),len(train1),len(test0),len(test1)])

print('Your trainset and testset are generated successfully!')
pass

train0_list=[]
train1_list=[]
test0_list=[]
test1_list=[]

import pandas as pd
train0_stats_df = pd.DataFrame(columns = ['Mean', 'Dev']) 
train1_stats_df = pd.DataFrame(columns = ['Mean', 'Dev']) 
test0_stats_df = pd.DataFrame(columns = ['Mean', 'Dev']) 
test1_stats_df = pd.DataFrame(columns = ['Mean', 'Dev']) 




for x in train0:
    train0_list.append(x)

for x in train1:
    train1_list.append(x)

for x in test0:
    test0_list.append(x)

for x in test1:
    test1_list.append(x)
    
for y in train0_list:
    newrow={'Mean':numpy.mean(y), 'Dev':numpy.std(y) }
    train0_stats_df = train0_stats_df.append(newrow, ignore_index=True)

for y in train1_list:
    newrow={'Mean':numpy.mean(y), 'Dev':numpy.std(y) }
    train1_stats_df = train1_stats_df.append(newrow, ignore_index=True)

for y in test0_list:
    newrow={'Mean':numpy.mean(y), 'Dev':numpy.std(y) }
    test0_stats_df = test0_stats_df.append(newrow, ignore_index=True)

for y in test1_list:
    newrow={'Mean':numpy.mean(y), 'Dev':numpy.std(y) }
    test1_stats_df = test1_stats_df.append(newrow, ignore_index=True)

mean_feat1_digit0 = train0_stats_df['Mean'].mean()
var_feat1_digit0 = train0_stats_df['Mean'].var()

mean_feat2_digit0 = train0_stats_df['Dev'].mean()
var_feat2_digit0 = train0_stats_df['Dev'].var()

mean_feat1_digit1 = train1_stats_df['Mean'].mean()
var_feat1_digit1 = train1_stats_df['Mean'].var()

mean_feat2_digit1 = train1_stats_df['Dev'].mean()
var_feat2_digit1 = train1_stats_df['Dev'].var()

print('mean_feat1_digit0 {}'.format(mean_feat1_digit0))
print('var_feat1_digit0 {}'.format(var_feat1_digit0)) 
print('mean_feat2_digit0 {}'.format(mean_feat2_digit0)) 
print('var_feat2_digit0 {}'.format(var_feat2_digit0)) 
print('mean_feat1_digit1 {}'.format(mean_feat1_digit1)) 
print('var_feat1_digit1 {}'.format(var_feat1_digit1))
print('mean_feat2_digit1 {}'.format(mean_feat2_digit1))
print('var_feat2_digit1 {}'.format(var_feat2_digit1))

train0_features_df = pd.DataFrame(columns = ['mean_feat1_digit0', 'var_feat1_digit0','mean_feat2_digit0','var_feat2_digit0'])
newrow={'mean_feat1_digit0':mean_feat1_digit0,
        'var_feat1_digit0':var_feat1_digit0,
        'mean_feat2_digit0':mean_feat2_digit0,
        'var_feat2_digit0':var_feat2_digit0}
train0_features_df = train0_features_df.append(newrow, ignore_index=True)

train1_features_df = pd.DataFrame(columns = ['mean_feat1_digit1', 'var_feat1_digit1','mean_feat2_digit1','var_feat2_digit1'])
newrow={'mean_feat1_digit1':mean_feat1_digit1,
        'var_feat1_digit1':var_feat1_digit1,
        'mean_feat2_digit1':mean_feat2_digit1,
        'var_feat2_digit1':var_feat2_digit1}
train1_features_df = train1_features_df.append(newrow, ignore_index=True)

def gp(x, mean_y, variance_y):

    # Input the arguments into a probability density function
    p = 1/(numpy.sqrt(2*numpy.pi*variance_y)) * numpy.exp((-(x-mean_y)**2)/(2*variance_y))
    
    # return p
    return p

correct = 0

for index, row in test0_stats_df.iterrows(): 
    #print (row["Mean"], row["Dev"])
    probability0 = 0
    probability1 = 0
    
    probability0 = (gp(row['Mean'], train0_features_df['mean_feat1_digit0'].values[0],train0_features_df['var_feat1_digit0'].values[0]) *
        gp(row['Dev'], train0_features_df['mean_feat2_digit0'].values[0],train0_features_df['var_feat2_digit0'].values[0]) * .5)
    
    probability1 = (gp(row['Mean'], train1_features_df['mean_feat1_digit1'].values[0],train1_features_df['var_feat1_digit1'].values[0]) *
        gp(row['Dev'], train1_features_df['mean_feat2_digit1'].values[0], train1_features_df['var_feat2_digit1'].values[0])* .5)
    
    #print('probability of 0: {} probability of 1: {}'.format(probability0, probability1))
    
    if probability0 > probability1:
        correct = correct + 1
    
print('total correct 0'': {}'.format(correct/len(test0_stats_df)))

correct = 0

for index, row in test1_stats_df.iterrows(): 
    #print (row["Mean"], row["Dev"])
    probability0 = 0
    probability1 = 0
    
    probability0 = (gp(row['Mean'], train0_features_df['mean_feat1_digit0'].values[0],train0_features_df['var_feat1_digit0'].values[0]) *
        gp(row['Dev'], train0_features_df['mean_feat2_digit0'].values[0],train0_features_df['var_feat2_digit0'].values[0]) * .5)
    
    probability1 = (gp(row['Mean'], train1_features_df['mean_feat1_digit1'].values[0],train1_features_df['var_feat1_digit1'].values[0]) *
        gp(row['Dev'], train1_features_df['mean_feat2_digit1'].values[0],train1_features_df['var_feat2_digit1'].values[0]) * .5)
    
    #print('probability of 0: {} probability of 1: {}'.format(probability0, probability1))
    
    if probability1 > probability0:
        correct = correct + 1
    
print('total correct 1'': {}'.format(correct/len(test1_stats_df)))

