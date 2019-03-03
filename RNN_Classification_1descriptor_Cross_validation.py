# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:39:33 2019

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

#Choose on of the 5 descriptors for the RNN Classification
Name_descriptors = ["WangSignaturesCEDD", "WangSignaturesFCTH", "WangSignaturesFuzzyColorHistogr", "WangSignaturesJCD", "WangSignaturesPHOG"]
Number_descriptor = 4 #The value to choose is beetwen [0:4]

#Get the descriptor length
length_dscpt = np.shape(descriptor_df.get(Name_descriptors[Number_descriptor]))[1]-1

#Define the model architecture for Not normalized data
model = Sequential()
model.add(Dense(64, input_dim=length_dscpt,activation = "relu")) #Layer 1 
model.add(Dense(64,activation = "relu")) #Layer 2
model.add(Dense(32,activation = "relu")) #Layer 3
model.add(Dense(32,activation = "relu")) #Layer 4
model.add(Dense(32,activation = "relu")) #Layer 5
model.add(Dense(16,activation = "relu")) #Layer 6
model.add(Dense(16,activation = "relu")) #Layer 7
model.add(Dense(16,activation = "relu")) #Layer 8
model.add(Dense(Nb_g,activation = "sigmoid")) #Output Layer
#Configure the model
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

#Define the model architecture for normalized data
model_norm = Sequential()
model_norm.add(Dense(64, input_dim=length_dscpt,activation = "relu")) #Layer 1 
model_norm.add(Dense(64,activation = "relu")) #Layer 2
model_norm.add(Dense(32,activation = "relu")) #Layer 3
model_norm.add(Dense(32,activation = "relu")) #Layer 4
model_norm.add(Dense(32,activation = "relu")) #Layer 5
model_norm.add(Dense(16,activation = "relu")) #Layer 6
model_norm.add(Dense(16,activation = "relu")) #Layer 7
model_norm.add(Dense(16,activation = "relu")) #Layer 8
model_norm.add(Dense(Nb_g,activation = "sigmoid")) #Output Layer
#Configure the model
model_norm.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
        
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


#for it_cr_val in range(1):
for it_cr_val in range(len(iter_cross_validation)):
    
    #Split the data set into train, validation and test sets
    indice_train = [] #List of images index for training set
    indice_val = [] #List of images index for validation set
    indice_test = [] #List of images index for test set
    
    #Setting up the index images for the 3 sets
    for l in range(0, 600): indice_train.append(int(iter_cross_validation.iloc[it_cr_val][l]))
    for m in range(600, 800): indice_val.append(int(iter_cross_validation.iloc[it_cr_val][m]))
    for n in range(800, 1000): indice_test.append(int(iter_cross_validation.iloc[it_cr_val][n]))
    
    #Setting up the train set descriptors normalize and Not normalize
    x_train_norm = np.zeros((len(indice_train), length_dscpt))
    x_train = np.zeros((len(indice_train), length_dscpt))

    y_train = np.zeros((len(indice_train), Nb_g))
    y1_train = np.zeros((1,len(indice_train)))
    
    y1_train[0,:] =  iter_cross_validation_label.iloc[it_cr_val][0:600]
    y1_train = np.transpose(y1_train)
    y1_train = y1_train.astype(int)
    
    for i in range(len(indice_train)):
        y_train[i,y1_train[i,0]] = 1

    compt = 0
    for idx in indice_train:
    
        # get the index of the image unknown
        index_img = np.where(df_file[0][:] == str(idx) + '.jpg')
        index_img = int(index_img[0])
    
        #Set up descriptor (Norm and NoNorm) for all the images
        descriptor_img = np.zeros((1,length_dscpt))
        descriptor_norm_img = np.zeros((1,length_dscpt))
        
        x_train[compt,:] = descriptor_df.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
        x_train_norm[compt,:] = descriptor_df_norm.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
        compt = compt + 1

    #Setting up the validation set descriptors normalize and Not normalize
    x_val_norm = np.zeros((len(indice_val), length_dscpt))
    x_val = np.zeros((len(indice_val), length_dscpt))
    
    y_val = np.zeros((len(indice_val), Nb_g))
    y1_val = np.zeros((1,len(indice_val)))
    
    y1_val[0,:] =  iter_cross_validation_label.iloc[it_cr_val][600:800]
    y1_val = np.transpose(y1_val)
    y1_val = y1_val.astype(int)
    
    for i in range(len(indice_val)):
        y_val[i,y1_val[i,0]] = 1
    
    compt = 0
    for idx in indice_val:
        
        # get the index of the image unknown
        index_img = np.where(df_file[0][:] == str(idx) + '.jpg')
        index_img = int(index_img[0])
        
        #Concatenate all the descriptors (Norm and NoNorm) of image Nb (unknown)
        descriptor_img = np.zeros((1,length_dscpt))
        descriptor_norm_img = np.zeros((1,length_dscpt))
        
        x_val[compt,:] = descriptor_df.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
        x_val_norm[compt,:] = descriptor_df_norm.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
        compt = compt + 1
        
    #Setting up the test set descriptors normalize and Not normalize
    x_test_norm = np.zeros((len(indice_test), length_dscpt))
    x_test = np.zeros((len(indice_test), length_dscpt))
    
    y_test = np.zeros((len(indice_test), Nb_g))
    y1_test = np.zeros((1,len(indice_test)))
    
    y1_test[0,:] =  iter_cross_validation_label.iloc[it_cr_val][800:1000]
    y1_test = np.transpose(y1_test)
    y1_test = y1_test.astype(int)
    
    for i in range(len(indice_test)):
        y_test[i,y1_test[i,0]] = 1
    
    compt = 0
    for idx in indice_test:
        
        # get the index of the image unknown
        index_img = np.where(df_file[0][:] == str(idx) + '.jpg')
        index_img = int(index_img[0])
        
        #Set up descriptors (Norm and NoNorm) of all the images
        descriptor_img = np.zeros((1,length_dscpt))
        descriptor_norm_img = np.zeros((1,length_dscpt))
        
        x_test[compt,:] = descriptor_df.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
        x_test_norm[compt,:] = descriptor_df_norm.get(Name_descriptors[Number_descriptor]).loc[index_img][1:]
        compt = compt + 1
                
    #Learning for Not normalized descriptors
    model.fit(x_train, y_train, batch_size=20, epochs=65, validation_data=(x_val,y_val))
    
    #Test
    Y_pred = model.predict(x_test) 
    Y_pred1 = np.copy(Y_pred)
    for i in range(len(indice_test)):
        coord = int(np.where(Y_pred[i,:] == np.amax(Y_pred[i,:]))[0][0])
        Y_pred[i,:] = 0
        Y_pred[i,coord] = 1
        
    
    Y_predL = np.zeros((len(indice_test), 1))
    
    for i in range(len(indice_test)):
        Y_predL[i,0] = int(np.where(Y_pred[i,:] == 1)[0][0])
    
    
    results.append(confusion_matrix(y1_test, Y_predL))
    #print(results)
    
    #Learning for normalized descriptors
    model_norm.fit(x_train_norm, y_train, batch_size=20, epochs=65, validation_data=(x_val_norm,y_val))
    
    #Test
    Y_pred_norm = model_norm.predict(x_test_norm) 
    
    for i in range(len(indice_test)):
        coord = int(np.where(Y_pred_norm[i,:] == np.amax(Y_pred_norm[i,:]))[0][0])
        Y_pred_norm[i,:] = 0
        Y_pred_norm[i,coord] = 1
        
    
    Y_predL_norm = np.zeros((len(indice_test), 1))
    
    for i in range(len(indice_test)):
        Y_predL_norm[i,0] = int(np.where(Y_pred_norm[i,:] == 1)[0][0])
    
    
    results_norm.append(confusion_matrix(y1_test, Y_predL_norm))
    #print(results_norm)
    
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
plt.title("Confusion matrix Sum RNN descriptor (" +  Name_descriptors[Number_descriptor] +  ") Not normalized")
sn.heatmap(results_sum, annot=True, cmap="YlGnBu", linewidths=5.0)

plt.figure(figsize = (10,10))
plt.title("Confusion matrix Sum RNN descriptor (" +  Name_descriptors[Number_descriptor] +  ") normalized")
sn.heatmap(results_sum_norm, annot=True, cmap="YlGnBu", linewidths=5.0)

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

x_clas = np.zeros((1,length_dscpt))
x_clas_norm = np.zeros((1,length_dscpt))    


x_clas[0,:] = descriptor_df.get(Name_descriptors[Number_descriptor]).loc[index_image_clas][1:]
x_clas_norm[0,:] = descriptor_df_norm.get(Name_descriptors[Number_descriptor]).loc[index_image_clas][1:]

Y_clas = model.predict(x_clas)
Y_clas_norm = model.predict(x_clas_norm)

coord = int(np.where(Y_clas[0,:] == np.amax(Y_clas[0,:]))[0][0])
Y_clas[0,:] = 0
Y_clas[0,coord] = 1

coord_norm = int(np.where(Y_clas_norm[0,:] == np.amax(Y_clas_norm[0,:]))[0][0])
Y_clas_norm[0,:] = 0
Y_clas_norm[0,coord_norm] = 1

print("The group of image " + str(image_clas) + ".jpg" + " : " + name_group[coord] + " (Not normalized classification)")
print("The group of image " + str(image_clas) + ".jpg" + " : " + name_group[coord] + " (normalized classification)")

plt.figure()
plt.title("Image Unknown (" + str(image_clas) + ".jpg)")
img = cv2.imread('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/Wang/' + str(image_clas) + '.jpg')
plt.xlabel("Not normalized classification --> " + "(" + name_group[coord] + ")" + " normalized classification --> " + "(" + name_group[coord_norm] + ")")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
