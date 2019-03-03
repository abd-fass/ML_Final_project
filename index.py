# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:46:36 2019

@author: FMA
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

#Read the data file that contain the image's descriptors
df_file = pd.read_excel('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures.xls', header=None)
df = pd.ExcelFile('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures.xls', header=None)

#Get the descriptors sheet from the xls file 
descriptor_df = {}
for sheet_name in df.sheet_names:
    descriptor_df[sheet_name] = df.parse(sheet_name, header=None)
    

""" Indexation """

#Get the Number of images
length_img = len(df_file)

#Get the unknown image for the indexation randomly
#Nb_img_rand = np.random.randint(length_img)
Nb_img_rand = 570
# get the index of the image unknown
index_img = np.where(df_file[0][:] == str(Nb_img_rand) + '.jpg')
index_img = int(index_img[0])


#Get all descriptor length
length_dscpt = 0
for dscpt in descriptor_df:
    descriptor = descriptor_df.get(dscpt)
    length_dscpt = length_dscpt + np.shape(descriptor)[1]-1
    
#Concatenate all the descriptors of image Nb (unknown)
descriptor_Nb_img = np.zeros((1,length_dscpt))
size_concat = 0 
for dscpt in descriptor_df:
    descriptor = descriptor_df.get(dscpt)
    descriptor_Nb_img[0,size_concat:size_concat+np.shape(descriptor)[1]-1] = descriptor.loc[index_img][1:]
    size_concat = size_concat + np.shape(descriptor)[1]-1



#vector that contain the distance between all the images and the image (unknown) choosed for the indexation
dist = np.zeros((1,length_img - 1))

#vector that contain the names of all the images compute for the distance calculation
index = []

#The loop for the indexation calculation
compt = 0
for i in range(length_img):
    
    # get the name of the image i
    name_img = df_file[0][i]
    
    #Avoid compute the calculation with image Nb (unknown)
    if (name_img != str(Nb_img_rand) + '.jpg'):
        #get all the descriptor for image i and concatenate them
        descriptor_img = np.zeros((1,length_dscpt))
        size_concat1 = 0 
        for dscpt1 in descriptor_df:
            descriptor1 = descriptor_df.get(dscpt1)
            descriptor_img[0,size_concat1:size_concat1 + np.shape(descriptor1)[1]-1] = descriptor1.loc[i][1:]
            size_concat1 = size_concat1 + np.shape(descriptor1)[1] - 1
        
        #Calculate the distance between image i and image_NB (unknown)
        sum_val = 0
        for j in range(length_dscpt):
            
            sum_val = sum_val + ( ( descriptor_Nb_img[0,j] - descriptor_img[0,j] )**2 )
        
        dist[0,compt] = np.sqrt(sum_val)
        compt = compt + 1
        
        #save the name of the image i in the same column as the dist value
        index.append(name_img)
        
    print(round(((i/(length_img-1))*100), 2), '% Done (indexation)')

            

    
""" Get Images indexation  """

#lists that contain the value of rows and cols for subplot depending on N_index choosed

#List of N_index + img index  (there will be N_index image + the unknown image to plot)
list_N_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#List of rows subplot
list_row_subplot = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]
#List of cols subplot
list_col_subplot = [1,2,3,4,3,3,4,4,3,4,4,4,4,4,4,4,4,4,4,4]

#Number of images indexation
N_index = 5
#Get the minimum dist
minimum_indexes= dist.argsort()

row = list_row_subplot[N_index]
col = list_col_subplot[N_index]

#Display the unknown image
plt.figure()
plt.subplot(row,col,1)
plt.title("Image Unknown (" + str(Nb_img_rand) + ".jpg)")
img = cv2.imread('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/Wang/' + str(Nb_img_rand) + '.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

for plot_img in range(N_index):
    
    #Display the index image number (plot_img+1)    
    plt.subplot(row,col,(plot_img+2))
    plt.title("Image index " + str(plot_img+1) + " (" + str(index[minimum_indexes[0,plot_img]]) + ")")
    img = cv2.imread('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/Wang/' + str(index[minimum_indexes[0,plot_img]]))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()   
