# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:05:50 2019

@author: FMA
"""

import numpy as np
import pandas as pd
from random import shuffle


""" Read the data """
#Read the data file that contain the image's descriptors
df_file = pd.read_excel('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures.xls', header=None)

    
length_img = np.shape(df_file)[0]
length_s_g = 20

""" Setting Up the cross validation """

#Create randomly list of images index

list_index_random = [i for i in range(0, length_img)] #list that contain the random images index 
shuffle(list_index_random)
list_label_random = [] #list that contain the labeling images depending on theres groups

#Generate the labeling list

for index in list_index_random:       
    #Labeling process
    if( (index>=0) and (index<=99) ) : list_label_random.append(0)
    elif( (index>=100) and (index<=199) ) : list_label_random.append(1)
    elif( (index>=200) and (index<=299) ) : list_label_random.append(2)
    elif( (index>=300) and (index<=399) ) : list_label_random.append(3)
    elif( (index>=400) and (index<=499) ) : list_label_random.append(4)
    elif( (index>=500) and (index<=599) ) : list_label_random.append(5)
    elif( (index>=600) and (index<=699) ) : list_label_random.append(6)
    elif( (index>=700) and (index<=799) ) : list_label_random.append(7)
    elif( (index>=800) and (index<=899) ) : list_label_random.append(8)
    elif( (index>=900) and (index<=999) ) : list_label_random.append(9)
    

#Create the s groups for cross validation
s1 = list_index_random[0:200]
s2 = list_index_random[200:400]
s3 = list_index_random[400:600]
s4 = list_index_random[600:800]
s5 = list_index_random[800:1000]

s1_label = list_label_random[0:200]
s2_label = list_label_random[200:400]
s3_label = list_label_random[400:600]
s4_label = list_label_random[600:800]
s5_label = list_label_random[800:1000]


#Index cross validation
iter_cross_validation =[]
iter_cross_validation.append(s2 + s3 + s4 + s5 + s1)
iter_cross_validation.append(s1 + s3 + s5 + s4 + s2)
iter_cross_validation.append(s1 + s4 + s5 + s2 + s3)
iter_cross_validation.append(s1 + s2 + s5 + s3 + s4)
iter_cross_validation.append(s2 + s3 + s4 + s1 + s5)

#Label cross validation
iter_cross_validation_label =[]
iter_cross_validation_label.append(s2_label + s3_label + s4_label + s5_label + s1_label)
iter_cross_validation_label.append(s1_label + s3_label + s5_label + s4_label + s2_label)
iter_cross_validation_label.append(s1_label + s4_label + s5_label + s2_label + s3_label)
iter_cross_validation_label.append(s1_label + s2_label + s5_label + s3_label + s4_label)
iter_cross_validation_label.append(s2_label + s3_label + s4_label + s1_label + s5_label)


iter_cross_validation1 = np.zeros((5,1000))
iter_cross_validation1[0,:] = iter_cross_validation[0][:]
iter_cross_validation1[1,:] = iter_cross_validation[1][:]
iter_cross_validation1[2,:] = iter_cross_validation[2][:]
iter_cross_validation1[3,:] = iter_cross_validation[3][:]
iter_cross_validation1[4,:] = iter_cross_validation[4][:]


iter_cross_validation_label1 = np.zeros((5,1000))
iter_cross_validation_label1[0,:] = iter_cross_validation_label[0][:]
iter_cross_validation_label1[1,:] = iter_cross_validation_label[1][:]
iter_cross_validation_label1[2,:] = iter_cross_validation_label[2][:]
iter_cross_validation_label1[3,:] = iter_cross_validation_label[3][:]
iter_cross_validation_label1[4,:] = iter_cross_validation_label[4][:]

# Save the iter_cross_validation in csv files
df_iter_cross_validation = pd.DataFrame(iter_cross_validation1)
df_iter_cross_validation_label = pd.DataFrame(iter_cross_validation_label1)
df_iter_cross_validation.to_csv('iter_cross_validation.csv', index=False) 
df_iter_cross_validation_label.to_csv('iter_cross_validation_label.csv', index=False) 








