# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:13:17 2019

@author: FMA
"""

import numpy as np
import pandas as pd

#Read the data file that contain the image's descriptors
df_file = pd.read_excel('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures.xls', header=None)
df = pd.ExcelFile('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures.xls', header=None)

df_file_norm = df_file.copy()


#Get the descriptors sheet from the xls file 
descriptor_df = {}
for sheet_name in df.sheet_names:
    descriptor_df[sheet_name] = df.parse(sheet_name, header=None)

descriptor_df_norm = descriptor_df.copy()
    
#Get the discriptor one by one
compt = 1
for dscpt in descriptor_df:
    descriptor = descriptor_df.get(dscpt)
    
    #Loop to normalize the descriptor dscpt for all the images
    for i in range(len(descriptor)):
        
        #normalize the descriptor_img
        descriptor_normalize = np.zeros((1,(np.shape(descriptor)[1]-1)))
        for dscpt_i in range(np.shape(descriptor)[1]-1):
            descriptor_normalize[0,dscpt_i] = ( (descriptor.loc[i][dscpt_i+1] - np.amin(descriptor.loc[i][1:])) / (np.amax(descriptor.loc[i][1:]) - np.amin(descriptor.loc[i][1:])) )
            
        #save the descriptor_normalize in descriptor_df_norm
        descriptor_df_norm[dscpt].loc[i,1:] = descriptor_normalize[0,:]
        
        print(round(((compt/(len(descriptor)*(len(descriptor_df))))*100), 2), '% Done (Normalization dataFrame)')
        compt += 1
        
#Save the descriptor_df_norm on xls file
        
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('D:/IOI M2 2018-2019/01-Machine Learning/TPs/TP-05-/prog/WangGlobalDescr/WangSignatures_norm.xls', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
for dscpt_save in descriptor_df:
    descriptor_save = descriptor_df_norm.get(dscpt_save)
    descriptor_save.to_excel(writer, sheet_name=dscpt_save, header=False, index=False)
    
# Close the Pandas Excel writer and output the Excel file.
writer.save()


        

