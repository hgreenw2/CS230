import pandas as pd
import numpy as np
import os


from skimage.io import imread
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import csv, copy
from sklearn.model_selection import train_test_split
import cv2


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
        print(csv_reader)
        #for row in csv_reader:
        #    key = row['file name']
        #    val = row['defect status']
        #    Y[key] = val
        
    return(Y)


x = constructY(r'C:\Users\hgree\OneDrive\Documents\CS230\proj_data\train_binary.csv')
pic_list = list(x.keys())
num_pics = len(pic_list)
num_trains = round(0.8*num_pics)
num_tests = num_pics - num_trains

for i in range(num_trains):
    Train_dict[pic_list(i)] = Y[pic_list(i)]

for i in num_trains + range(num_tests):
    Test_dict[pic_list(i)] = Y[pic_list(i)]


##for name in x.keys():
##    in_img = name
##    if 1:
##        path = os.path.join(r'C:\Users\hgree\OneDrive\Documents\CS230\proj_data\train_images_square', str(in_img))
##        img = cv2.imread(path)
##          #get size
##        height, width, channels = img.shape
##          #print (in_img,height, width, channels)
##          # Create a black image
##        x = height if height > width else width
##        y = height if height > width else width
##        square= np.zeros((x,y,3), np.uint8)
##        square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
##        #resize to 256x256
##        #square = cv2.resize(square, (256, 256))
##        cv2.imwrite(path,square)
##        cv2.waitKey(0)
##

