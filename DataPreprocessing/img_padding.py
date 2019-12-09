#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import pandas as pd
import numpy as np
import os

# In[ ]:


from skimage.io import imread
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import csv, copy
from sklearn.model_selection import train_test_split


# In[ ]:


#function that takes as input a .csv file name (string) in the working directory 
#and outputs a dictionary with jpeg names for keys and numpy arrays of shape (4,)
#for values. These values correspond to whether or not a defect exists- defect 
#type corresponds to index of the array.

def constructY(filename):
    
    Y = {}  #the final dictionary we want- jpg names point to numpy array of 0s and 1s (which defects exist)
    defectvect = np.zeros(4)
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            #split the file name into jpeg name and defect type
            [img_name, defect_type] = row["ImageId_ClassId"].split('_')
            defect_type = int(defect_type)
        
            #get a boolean, 1 if defect exists, 0 if none exist
            defect_exist = int(row["EncodedPixels"] != "")
            #enter the 1 or 0 into the 4 element numpy array
            defectvect[defect_type - 1] = defect_exist
            line_count += 1

            if (defect_type == 4):  #when defect type is 4, we're at the last defect
                Y[img_name] = copy.deepcopy(defectvect) #we need to ensure we're assigning values to the variable, not a pointer
 
    for key in Y.keys():
        #print(f'{key} -> {Y[key]}')
    #print(f'Processed {line_count} lines.')
        None
        
    return(Y)


# #These are google colab file upload functions. Basically, Im uploading two things - 1. the train.csv file (to run your function on it) and 2. a zip containing all the images I want to train on
# #These two files need to be present in the root folder of the project that has the notebook. In jupyter I guess you could use the toolbar to directly upload these files without using a command.
# #I created a zip file with 1000 images called 'mini_train.zip' and will be using that from here on.

# In[ ]:


x = constructY(r'C:\Users\hgree\OneDrive\Documents\CS230\proj_data\train.csv')


# In[ ]:


imageCol = ['ImageId_ClassId']
imageNames = pd.read_csv(r'C:\Users\hgree\OneDrive\Documents\CS230\proj_data\train.csv', names=imageCol)


# In[ ]:


images = []
defects = []

#creating two lists - one for the list of image names and one for a list of binary values for each image - 0: no defect . 1: defect present


# In[ ]:


for key in x.keys():
  images.append(key)
  defectArray = x[key]
  defFlag = 0
  for i in defectArray:
    if i == 1:
      defFlag = 1
      break 
  if defFlag == 0:
    defects.append(0)
  else:
    defects.append(1)
    
#iterating to check if a defect is there. if any of the values in the x dict is 1, there is a defect.


import cv2

for name in x.keys():
    in_img = name
    if 1:
        path = os.path.join(r'C:\Users\hgree\OneDrive\Documents\CS230\proj_data\train_images_square', str(in_img))
        img = cv2.imread(path)
          #get size
        height, width, channels = img.shape
          #print (in_img,height, width, channels)
          # Create a black image
        x = height if height > width else width
        y = height if height > width else width
        square= np.zeros((x,y,3), np.uint8)
        square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
        #resize to 256x256
        #square = cv2.resize(square, (256, 256))
        cv2.imwrite(path,square)
        cv2.waitKey(0)

#using opencv to convert all the rectangular images to squares (by just padding black pixels around)


# In[ ]:


i = 0
plt.figure(figsize=(10,10))
img = imread('000a4bcdd.jpg')
#plt.imshow(img)
print('image shape is ', img.shape)

#to check how the squared image looks (size 1600 x 1600)
