# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 03:32:41 2019

@author: FMA
"""

import numpy as np
import pandas as pd
import cv2
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sn

""" Read the data """
#Read the data file that contain the image's descriptors
df_file = pd.read_excel('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures.xls', header=None)
df = pd.ExcelFile('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures.xls', header=None)

#Read the data file that contain the image's descriptors normalized
df_file_norm = pd.read_excel('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures_norm.xls', header=None)
df_norm = pd.ExcelFile('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures_norm.xls', header=None)

#Get the descriptors sheet from the xls file 
descriptor_df = {}
descriptor_df_norm = {}
for sheet_name in df.sheet_names:
    descriptor_df[sheet_name] = df.parse(sheet_name, header=None)
    descriptor_df_norm[sheet_name] = df_norm.parse(sheet_name, header=None)
    
""" Split the data into train, validation and test sets by applying cross validation"""
#Split randomly the database image into train, validation and test 
length_img = 1000 #Number of all images of all groups combined
length_s_g = 20 #Number of images in each s groups
Nb_g = 10 #Number of groups
length_g = 100 #Numbers of image in each group
length_g_train = 60 #Number of images (in each group) for training set
length_g_val = 20 #Number of images (in each group) for validation set
length_g_test = 20 #Number of images (in each group) for test set

#Setting Up the images index in a matrix of shape(length_g, Nb_g)
index_img_mat = np.zeros((length_g, Nb_g))
count = 0
for j in range(Nb_g):
    for i in range(length_g):
        index_img_mat[i,j] = count
        count += 1
    
#Create the vector s that contain the images index for cross validation 
s1 = [it for it in range(0, 200)]
s2 = [it for it in range(200, 400)]
s3 = [it for it in range(400, 600)]
s4 = [it for it in range(600, 800)]
s5 = [it for it in range(800, 1000)]

#Cross validation table
iter_cross_validation =[]
iter_cross_validation.append([s2,s3,s4,s5,s1])
iter_cross_validation.append([s1,s3,s4,s5,s2])
iter_cross_validation.append([s1,s2,s4,s5,s3])
iter_cross_validation.append([s1,s2,s3,s5,s4])
iter_cross_validation.append([s1,s2,s3,s4,s5])

index_cross_validation = []
index_cross_validation.append([1,2,3,4,0])
index_cross_validation.append([0,2,3,4,1])
index_cross_validation.append([0,1,3,4,2])
index_cross_validation.append([0,1,2,4,3])
index_cross_validation.append([0,1,2,3,4])

#Get all descriptor length
length_dscpt = 0
for dscpt in descriptor_df:
    descriptor = descriptor_df.get(dscpt)
    length_dscpt = length_dscpt + np.shape(descriptor)[1]-1

#Define the model architecture for Not normalized data
model = Sequential()
model.add(Dense(64, input_dim=length_dscpt,activation = "relu")) #Layer 1 
model.add(Dense(64,activation = "relu")) #Layer 2
model.add(Dense(32,activation = "relu")) #Layer 3
model.add(Dense(Nb_g,activation = "sigmoid")) #Output Layer
#Configure the model
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


results = []
results_norm = []

accuracy = []
accuracy_norm = []

#for it_cr_val in range(1):
for it_cr_val in range(len(iter_cross_validation)):
    
    #Split the data set into train, validation and test sets
    indice_train = [] #List of images index for training set
    indice_val = [] #List of images index for validation set
    indice_test = [] #List of images index for test set
    
    #Setting up the index images for the 3 sets
    indice_train = iter_cross_validation[it_cr_val][0] + iter_cross_validation[it_cr_val][1] + iter_cross_validation[it_cr_val][2]
    indice_val = iter_cross_validation[it_cr_val][3]
    indice_test = iter_cross_validation[it_cr_val][4]
    
    #Setting up the train set descriptors normalize and Not normalize
    x_train_norm = np.zeros((len(indice_train), length_dscpt))
    x_train = np.zeros((len(indice_train), length_dscpt))

    y_train = np.zeros((len(indice_train), Nb_g))
    y1_train = np.zeros((len(indice_train), 1))

    compt = 0
    for idx in indice_train:
    
        # get the index of the image unknown
        index_img = np.where(df_file[0][:] == str(idx) + '.jpg')
        index_img = int(index_img[0])
    
        #Concatenate all the descriptors (Norm and NoNorm) for all the images
        descriptor_img = np.zeros((1,length_dscpt))
        descriptor_norm_img = np.zeros((1,length_dscpt))
        size_concat = 0 
        for dscpt in descriptor_df:
            descriptor = descriptor_df.get(dscpt)
            descriptor_norm = descriptor_df_norm.get(dscpt)
            descriptor_img[0,size_concat:size_concat+np.shape(descriptor)[1]-1] = descriptor.loc[index_img][1:]
            descriptor_norm_img[0,size_concat:size_concat+np.shape(descriptor_norm)[1]-1] = descriptor_norm.loc[index_img][1:]
            size_concat = size_concat + np.shape(descriptor)[1]-1
    
        x_train_norm[compt,:] = descriptor_norm_img[0,:]
        x_train[compt,:] = descriptor_img[0,:]
        compt = compt + 1
    
    
    for i in range(200):
        y_train[i,index_cross_validation[it_cr_val][0]] = 1
        y1_train[i,0] = index_cross_validation[it_cr_val][0]
    
    for i in range(200, 400):
        y_train[i,index_cross_validation[it_cr_val][1]] = 1
        y1_train[i,0] = index_cross_validation[it_cr_val][1]        

    for i in range(400, 600):
        y_train[i,index_cross_validation[it_cr_val][2]] = 1
        y1_train[i,0] = index_cross_validation[it_cr_val][2]    
    
    

    #Setting up the validation set descriptors normalize and Not normalize
    x_val_norm = np.zeros((len(indice_val), length_dscpt))
    x_val = np.zeros((len(indice_val), length_dscpt))
    
    y_val = np.zeros((len(indice_val), Nb_g))
    y1_val = np.zeros((len(indice_val), 1))
    
    compt = 0
    for idx in indice_val:
        
        # get the index of the image unknown
        index_img = np.where(df_file[0][:] == str(idx) + '.jpg')
        index_img = int(index_img[0])
        
        #Concatenate all the descriptors (Norm and NoNorm) of image Nb (unknown)
        descriptor_img = np.zeros((1,length_dscpt))
        descriptor_norm_img = np.zeros((1,length_dscpt))
        size_concat = 0 
        for dscpt in descriptor_df:
            descriptor = descriptor_df.get(dscpt)
            descriptor_norm = descriptor_df_norm.get(dscpt)
            descriptor_img[0,size_concat:size_concat+np.shape(descriptor)[1]-1] = descriptor.loc[index_img][1:]
            descriptor_norm_img[0,size_concat:size_concat+np.shape(descriptor_norm)[1]-1] = descriptor_norm.loc[index_img][1:]
            size_concat = size_concat + np.shape(descriptor)[1]-1
        
        x_val_norm[compt,:] = descriptor_norm_img[0,:]
        x_val[compt,:] = descriptor_img[0,:]
        compt = compt + 1

    for i in range(0, 200):
        y_val[i,index_cross_validation[it_cr_val][3]] = 1
        y1_val[i,0] = index_cross_validation[it_cr_val][3] 
    
        
    #Setting up the test set descriptors normalize and Not normalize
    x_test_norm = np.zeros((len(indice_test), length_dscpt))
    x_test = np.zeros((len(indice_test), length_dscpt))
    
    y_test = np.zeros((len(indice_test), Nb_g))
    y1_test = np.zeros((len(indice_test), 1))
    
    compt = 0
    for idx in indice_test:
        
        # get the index of the image unknown
        index_img = np.where(df_file[0][:] == str(idx) + '.jpg')
        index_img = int(index_img[0])
        
        #Concatenate all the descriptors (Norm and NoNorm) of image Nb (unknown)
        descriptor_img = np.zeros((1,length_dscpt))
        descriptor_norm_img = np.zeros((1,length_dscpt))
        size_concat = 0 
        for dscpt in descriptor_df:
            descriptor = descriptor_df.get(dscpt)
            descriptor_norm = descriptor_df_norm.get(dscpt)
            descriptor_img[0,size_concat:size_concat+np.shape(descriptor)[1]-1] = descriptor.loc[index_img][1:]
            descriptor_norm_img[0,size_concat:size_concat+np.shape(descriptor_norm)[1]-1] = descriptor_norm.loc[index_img][1:]
            size_concat = size_concat + np.shape(descriptor)[1]-1
        
        x_test_norm[compt,:] = descriptor_norm_img[0,:]
        x_test[compt,:] = descriptor_img[0,:]
        compt = compt + 1
    
    for i in range(0, 200):
        y_test[i,index_cross_validation[it_cr_val][4]] = 1
        y1_test[i,0] = index_cross_validation[it_cr_val][4]
        
    
    #Learning for Not normalized descriptors
    model.fit(x_train, y_train, batch_size=20, epochs=65, validation_data=(x_val,y_val))
    
    #Test
    Y_pred = model.predict(x_test) 
    Y_pred1 = Y_pred
    for i in range(len(indice_test)):
        coord = int(np.where(Y_pred[i,:] == np.amax(Y_pred[i,:]))[0][0])
        Y_pred[i,:] = 0
        Y_pred[i,coord] = 1
        
    
    Y_predL = np.zeros((len(indice_test), 1))
    
    for i in range(len(indice_test)):
        Y_predL[i,0] = int(np.where(Y_pred[i,:] == 1)[0][0])
    
    
    results.append(confusion_matrix(y1_test, Y_predL))

    
#    acc = 0
#    for i in range(Nb_g):
#        acc = acc + results[it_cr_val][i,i]
#    acc = (acc/(length_g_test*Nb_g))*100
#    accuracy.append(acc)
    
    

""" Result """
plt.figure(figsize = (10,10))
plt.title("Confusion matrix RNN all descriptors Not normalized")
sn.heatmap(results[4], annot=True, cmap="YlGnBu", linewidths=5.0)

#
#print("accuracy = ", accuracy[4], "%")

