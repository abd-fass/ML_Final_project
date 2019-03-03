# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:21:45 2019

@author: FMA
"""

import numpy as np
import pandas as pd
import cv2
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
    

length_img = np.shape(df_file)[0]
length_s_g = 20
Nb_g = 10
length_g_test = 20 #Number of images (in each group) for test set

""" Load the cross validation """
iter_cross_validation = pd.read_csv('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/iter_cross_validation.csv')
iter_cross_validation_label = pd.read_csv('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/iter_cross_validation_label.csv')

""" Learning process """

K = 15 #Numbers of neighboors for the KNN algorithm

#Get all descriptor length
length_dscpt = 0
for dscpt in descriptor_df:
    descriptor = descriptor_df.get(dscpt)
    length_dscpt = length_dscpt + np.shape(descriptor)[1]-1
    
#Variable for statistics to calculate the efficiency of the algorithm
results = []
accuracy = []
Correct_sum = 0
Rong_sum = 0
accuracy_final = 0

results_norm = []
accuracy_norm = []
Correct_sum_norm = 0
Rong_sum_norm = 0
accuracy_final_norm = 0



for it_cr_val in range(len(iter_cross_validation)):
    
    #Split the data set into train, validation and test sets
    indice_train = [] #List of images index for training set
    #indice_val = [] #List of images index for validation set
    indice_test = [] #List of images index for test set
    
    #Setting up the index images for the 3 sets
    for l in range(0, 800): indice_train.append(int(iter_cross_validation.iloc[it_cr_val][l]))
    for n in range(800, 1000): indice_test.append(int(iter_cross_validation.iloc[it_cr_val][n]))
    
    #Setting up the train set descriptors normalize and Not normalize
    x_train_norm = np.zeros((len(indice_train), length_dscpt))
    x_train = np.zeros((len(indice_train), length_dscpt))

    #y_train = np.zeros((len(indice_train), Nb_g))
    y1_train = np.zeros((1,len(indice_train)))
    
    y1_train[0,:] =  iter_cross_validation_label.iloc[it_cr_val][0:800]
    y1_train = np.transpose(y1_train)
    y1_train = y1_train.astype(int)

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

        
    #Setting up the test set descriptors normalize and Not normalize
    x_test_norm = np.zeros((len(indice_test), length_dscpt))
    x_test = np.zeros((len(indice_test), length_dscpt))
    
    y1_test = np.zeros((1,len(indice_test)))
    
    y1_test[0,:] =  iter_cross_validation_label.iloc[it_cr_val][800:1000]
    y1_test = np.transpose(y1_test)
    y1_test = y1_test.astype(int)
    
    
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
    
    #KNN Algorithm 
    Y_predL = np.zeros((len(indice_test), 1)) #Predeciton vector Not normalized data
    Y_predL_norm = np.zeros((len(indice_test), 1)) #Predeciton vector Normalized data
    
    #Loop for all the images of test set
    for nb_test in range(len(indice_test)):
        
        dist_test = [] #List for the distance calculation (Not Normalized)
        vec_vote = np.zeros((Nb_g)) #Vector for the vote calculation (Not Normalized)

        dist_test_norm = [] #List for the distance calculation (Normalized)
        list_indice_K_norm = [] #List for the index of K neighboors (Normalized)
        label_K_norm = [] # list of the groups labels of each neigboors (Normalized)
        vec_vote_norm = np.zeros((Nb_g)) #Vector for the vote calculation (Normalized)
        
        #Loop for compute all the train images with nb_test image
        for nb_img_train in range(len(indice_train)):
            dist_temp = 0 #Variable tempo for the distance calculation (Not Normalized)
            dist_temp_norm = 0 #Variable tempo for the distance calculation (Normalized)
            
            #Loop for the distance calculation for the descriptors
            for nb_dscpt in range(length_dscpt): 
                
                #Distance calculation
                dist_temp = dist_temp + (x_train[nb_img_train,nb_dscpt] - x_test[nb_test,nb_dscpt])**2
                dist_temp_norm = dist_temp_norm + (x_train_norm[nb_img_train,nb_dscpt] - x_test_norm[nb_test,nb_dscpt])**2
            
            #Appending the distnace result with nb_test image and nb_img_train image
            dist_test.append(np.sqrt(dist_temp))
            dist_test_norm.append(np.sqrt(dist_temp_norm))
            
            #Max value of dist_test to remplace each k min 
            MAX = np.amax(dist_test)
            MAX_norm = np.amax(dist_test_norm)
            #Get the K neignboors
        for k in range(K):
                
            indice = dist_test.index(min((dist_test)))
            indice_norm = dist_test_norm.index(min((dist_test_norm)))
            vec_vote[y1_train[indice,0]] = vec_vote[y1_train[indice,0]] + 1 
            vec_vote_norm[y1_train[indice_norm,0]] = vec_vote_norm[y1_train[indice_norm,0]] + 1
            dist_test[indice] = MAX
            dist_test_norm[indice_norm] = MAX_norm
        
        #The predected classification
        Y_predL[nb_test,0] = np.argmax(vec_vote)
        Y_predL_norm[nb_test,0] = np.argmax(vec_vote_norm)
        
        print('\r (KNN Cross Validation ' + str(it_cr_val+1) + '/' + str(len(iter_cross_validation)) + ') Done '.format(round(((nb_test/(len(indice_test)-1))*100), 2))+str(round(((nb_test/(len(indice_test)-1))*100), 2)) + "%", end="")
        
    results.append(confusion_matrix(y1_test, Y_predL))
    results_norm.append(confusion_matrix(y1_test, Y_predL_norm))

    #Statistic calculation
    acc = 0
    for i in range(Nb_g):
        acc = acc + results[it_cr_val][i,i]
        Correct_sum = Correct_sum + results[it_cr_val][i,i]
        Rong_sum = Rong_sum + (length_s_g - results[it_cr_val][i,i])
    acc = (acc/(length_g_test*Nb_g))*100
    #print("accuracy = ", acc, "%")
    accuracy.append(acc)
     
        
    acc_norm = 0
    for i in range(Nb_g):
        acc_norm = acc_norm + results_norm[it_cr_val][i,i]
        Correct_sum_norm = Correct_sum_norm + results_norm[it_cr_val][i,i]
        Rong_sum_norm = Rong_sum_norm + (length_s_g - results_norm[it_cr_val][i,i])
    acc_norm = (acc_norm/(length_g_test*Nb_g))*100
    #print("accuracy_norm = ", acc_norm, "%")
    accuracy_norm.append(acc_norm)           


accuracy_final = (Correct_sum/length_img)*100
accuracy_final_norm = (Correct_sum_norm/length_img)*100   

""" Result """

results_sum = np.zeros((Nb_g,Nb_g))
results_sum_norm = np.zeros((Nb_g,Nb_g))

for iter in range(len(iter_cross_validation)):
    results_sum[:,:] = results_sum[:,:] + results[0][:,:]
    results_sum_norm[:,:] = results_sum_norm[:,:] + results_norm[0][:,:]
    

plt.figure(figsize = (10,10))
plt.title("Confusion matrix Sum KNN all descriptors Not normalized")
sn.heatmap(results_sum, annot=True, cmap="YlGnBu", linewidths=5.0)

plt.figure(figsize = (10,10))
plt.title("Confusion matrix Sum KNN all descriptors normalized")
sn.heatmap(results_sum_norm, annot=True, cmap="YlGnBu", linewidths=5.0)

print("___")
print("accuracy = ", accuracy[4], "%")
print("accuracy_norm = ", accuracy_norm[4], "%")

print("accuracy_final = ", accuracy_final, "%")
print("accuracy_final_norm = ", accuracy_final_norm, "%")

print("Correct_sum = ", Correct_sum)
print("Rong_sum = ", Rong_sum)

print("Correct_sum_norm = ", Correct_sum_norm)
print("Rong_sum_norm = ", Rong_sum_norm)


""" Classification """

name_group = ["Jungle", "Beach", "Monuments", "Bus", "Dinosaurs", "Elephants", "Flowers", "Horses", "Mountains", "Courses"]

#Image to classify
image_clas = np.random.randint(length_img)
#image_clas = 950
index_image_clas = np.where(df_file[0][:] == str(image_clas) + '.jpg')
index_image_clas = int(index_image_clas[0])

#Concatenate all the descriptors (Norm and NoNorm) for all the images
descriptor_image_clas = np.zeros((1,length_dscpt))
descriptor_norm_image_clas = np.zeros((1,length_dscpt))
size_concat = 0 
for dscpt in descriptor_df:
    descriptor = descriptor_df.get(dscpt)
    descriptor_norm = descriptor_df_norm.get(dscpt)
    descriptor_image_clas[0,size_concat:size_concat+np.shape(descriptor)[1]-1] = descriptor.loc[index_image_clas][1:]
    descriptor_norm_image_clas[0,size_concat:size_concat+np.shape(descriptor_norm)[1]-1] = descriptor_norm.loc[index_image_clas][1:]
    size_concat = size_concat + np.shape(descriptor)[1]-1

x_clas = np.zeros((1,length_dscpt))
x_clas_norm = np.zeros((1,length_dscpt))    
x_clas[0,:] = descriptor_image_clas[0,:]
x_clas_norm[0,:] = descriptor_norm_image_clas[0,:]

indice_set = []

for l in range(0, 1000): indice_set.append(int(iter_cross_validation.iloc[it_cr_val][l]))

x_set = np.zeros((len(indice_set),length_dscpt))
x_set_norm = np.zeros((len(indice_set),length_dscpt))

compt = 0
for idx in indice_set:
    
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
    
    x_set_norm[compt,:] = descriptor_norm_img[0,:]
    x_set[compt,:] = descriptor_img[0,:]
    compt = compt + 1


y1_set = np.zeros((1,len(indice_set)))
y1_set[0,:] =  iter_cross_validation_label.iloc[it_cr_val][0:1000]
y1_set = np.transpose(y1_set)
y1_set = y1_set.astype(int)
    
#KNN Algorithm 
dist_set = [] #List for the distance calculation (Not Normalized)
vec_vote = np.zeros((Nb_g)) #Vector for the vote calculation (Not Normalized)
    
dist_set_norm = [] #List for the distance calculation (Normalized)
vec_vote_norm = np.zeros((Nb_g)) #Vector for the vote calculation (Normalized)
#Loop for all the images of test set
for nb_set in range(len(indice_set)):

    #Avoid applying the KNN on the image that we want to classify
    if (indice_set[nb_set] != image_clas):

            
            
            
        dist_temp = 0 #Variable tempo for the distance calculation (Not Normalized)
        dist_temp_norm = 0 #Variable tempo for the distance calculation (Normalized)        
        #Loop for the distance calculation for the descriptors
        for nb_dscpt in range(length_dscpt): 
                    
            #Distance calculation
            dist_temp = dist_temp + (x_set[nb_set,nb_dscpt] - x_clas[0,nb_dscpt])**2
            dist_temp_norm = dist_temp_norm + (x_set_norm[nb_set,nb_dscpt] - x_clas_norm[0,nb_dscpt])**2
                
            #Appending the distnace result with nb_test image and nb_img_train image
        dist_set.append(np.sqrt(dist_temp))
        dist_set_norm.append(np.sqrt(dist_temp_norm))
                
        #Max value of dist_test to remplace each k min 
        MAX = np.amax(dist_set)
        MAX_norm = np.amax(dist_set_norm)
    print('\r (KNN Classification) Done '.format(round(((nb_set/(len(indice_set)-1))*100), 2))+str(round(((nb_set/(len(indice_set)-1))*100), 2)) + "%", end="")
    
#Get the K neignboors
for k in range(K):
                    
    indice = dist_set.index(min((dist_set)))
    indice_norm = dist_set_norm.index(min((dist_set_norm)))
    vec_vote[y1_set[indice,0]] = vec_vote[y1_set[indice,0]] + 1 
    vec_vote_norm[y1_set[indice_norm,0]] = vec_vote_norm[y1_set[indice_norm,0]] + 1
    dist_set[indice] = MAX
    dist_set_norm[indice_norm] = MAX_norm
        
#The predected classification
Y_clas = np.argmax(vec_vote)
Y_clas_norm = np.argmax(vec_vote_norm)


print("The group of image " + str(image_clas) + ".jpg" + " : " + name_group[Y_clas] + " (Not normalized classification)")
print("The group of image " + str(image_clas) + ".jpg" + " : " + name_group[Y_clas_norm] + " (normalized classification)")

plt.figure()
plt.title("Image Unknown (" + str(image_clas) + ".jpg)")
img = cv2.imread('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/Wang/' + str(image_clas) + '.jpg')
plt.xlabel("Not normalized classification --> " + "(" + name_group[Y_clas] + ")" + " normalized classification --> " + "(" + name_group[Y_clas_norm] + ")")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))       
            
        
    
    
    
    
    
    
    
        
    
    
    