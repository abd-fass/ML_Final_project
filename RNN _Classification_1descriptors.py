# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:49:56 2019

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
    

""" Split the data into train, validation and test sets"""
#Split randomly the database image into train, validation and test 
Nb_g = 10 #Number of groups
length_g = 100 #Numbers of image in each group
length_g_train = 60 #Number of images (in each group) for training set
length_g_val = 20 #Number of images (in each group) for validation set
length_g_test = 20 #Number of images (in each group) for test set
indice_train = [] #List of images index for training set
indice_val = [] #List of images index for validation set
indice_test = [] #List of images index for test set


begin_array = 0
end_array = length_g - 1
for i in range(Nb_g):
    
    #Create randomly list of index for each groups
    list_g=[]
    while (len(list_g) != length_g):
        r=random.randint(begin_array,end_array)
        if r not in list_g: 
            list_g.append(r)
    
    #Define the index images for training set
    for tr in range(length_g_train): 
        indice_train.append(list_g[0])
        list_g.remove(list_g[0])
    
    #Define the index images for validation set
    for vl in range(length_g_val): 
        indice_val.append(list_g[0])
        list_g.remove(list_g[0])
    
    #Define the index images for test set
    for tst in range(length_g_test): 
        indice_test.append(list_g[0])
        list_g.remove(list_g[0])
    
    begin_array = begin_array + length_g
    end_array = end_array + length_g
    
#Choose on of the 5 descriptors for the RNN Classification
Name_descriptors = ["WangSignaturesCEDD", "WangSignaturesFCTH", "WangSignaturesFuzzyColorHistogr", "WangSignaturesJCD", "WangSignaturesPHOG"]
Number_descriptor = 0 #The value to choose is beetwen [0:4]

#Get the descriptor length
length_dscpt = np.shape(descriptor_df.get(Name_descriptors[Number_descriptor]))[1]-1

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
    
    #Get the descriptors (Norm and NoNorm) for all the images
    descriptor_img = np.zeros((1,length_dscpt))
    descriptor_norm_img = np.zeros((1,length_dscpt))
    
    x_train[compt,:] = descriptor_df.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
    x_train_norm[compt,:] = descriptor_df_norm.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
    compt = compt + 1

a = 0
b = length_g_train 
label = 0
for i in range(Nb_g):
    y_train[a:b,i] = 1
    y1_train[a:b,0] = label
    a = a + length_g_train
    b = b + length_g_train
    label = label + 1
    
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
    
    #Get the descriptors (Norm and NoNorm) for all the images
    descriptor_img = np.zeros((1,length_dscpt))
    descriptor_norm_img = np.zeros((1,length_dscpt))
    
    x_val[compt,:] = descriptor_df.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
    x_val_norm[compt,:] = descriptor_df_norm.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
    compt = compt + 1

a = 0
b = length_g_val 
label = 0
for i in range(Nb_g):
    y_val[a:b,i] = 1
    y1_train[a:b,0] = label
    a = a + length_g_val
    b = b + length_g_val
    label = label + 1

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
    
    #Get the descriptors (Norm and NoNorm) for all the images
    descriptor_img = np.zeros((1,length_dscpt))
    descriptor_norm_img = np.zeros((1,length_dscpt))
    
    x_test[compt,:] = descriptor_df.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
    x_test_norm[compt,:] = descriptor_df_norm.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
    compt = compt + 1

a = 0
b = length_g_test 
label = 0
for i in range(Nb_g):
    y_test[a:b,i] = 1
    y1_test[a:b,0] = label
    a = a + length_g_test
    b = b + length_g_test
    label = label + 1
    
# Recommendation 
#Execute the last part (Setting Up the sets) only once so you'll have approximately the same result in the learning process step
    
""" Learning process """
#Define the model architecture
model = Sequential()
model.add(Dense(64, input_dim=length_dscpt,activation = "relu")) #Layer 1 
model.add(Dense(64,activation = "relu")) #Layer 2
model.add(Dense(32,activation = "relu")) #Layer 3
model.add(Dense(16,activation = "relu")) #Layer 4
model.add(Dense(Nb_g,activation = "sigmoid")) #Output Layer
#Configure the model
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

#Learning for Not normalized descriptors
model.fit(x_train, y_train, batch_size=32, epochs=250, validation_data=(x_val,y_val))

#Test
Y_pred = model.predict(x_test) 

for i in range(len(indice_test)):
    coord = int(np.where(Y_pred[i,:] == np.amax(Y_pred[i,:]))[0])
    Y_pred[i,:] = 0
    Y_pred[i,coord] = 1
    

Y_predL = np.zeros((len(indice_test), 1))

for i in range(len(indice_test)):
    Y_predL[i,0] = int(np.where(Y_pred[i,:] == 1)[0])


results = confusion_matrix(y1_test, Y_predL)
#print(results)

#Learning for normalized descriptors
model.fit(x_train_norm, y_train, batch_size=32, epochs=250, validation_data=(x_val_norm,y_val))

#Test
Y_pred_norm = model.predict(x_test_norm) 

for i in range(len(indice_test)):
    coord = int(np.where(Y_pred_norm[i,:] == np.amax(Y_pred_norm[i,:]))[0])
    Y_pred_norm[i,:] = 0
    Y_pred_norm[i,coord] = 1
    

Y_predL_norm = np.zeros((len(indice_test), 1))

for i in range(len(indice_test)):
    Y_predL_norm[i,0] = int(np.where(Y_pred_norm[i,:] == 1)[0])


results_norm = confusion_matrix(y1_test, Y_predL_norm)
#print(results_norm)

accuracy = 0
for i in range(Nb_g):
    accuracy = accuracy + results[i,i]
accuracy = (accuracy/(length_g_test*Nb_g))*100
print("accuracy = ", accuracy, "%")

plt.figure(figsize = (10,10))
plt.title("Confusion matrix RNN all descriptors Not normalized")
sn.heatmap(results, annot=True, cmap="YlGnBu", linewidths=5.0)

accuracy_norm = 0
for i in range(Nb_g):
    accuracy_norm = accuracy_norm + results_norm[i,i]
accuracy_norm = (accuracy_norm/(length_g_test*Nb_g))*100
print("accuracy_norm = ", accuracy_norm, "%")

plt.figure(figsize = (10,10))
plt.title("Confusion matrix RNN all descriptors normalized")
sn.heatmap(results_norm, annot=True, cmap="YlGnBu", linewidths=5.0)


