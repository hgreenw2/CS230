#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


from skimage.io import imread
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import csv, numpy as np, copy
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


# #fil = files.upload()
# #fil2 = files.upload()
# 
# #These are google colab file upload functions. Basically, Im uploading two things - 1. the train.csv file (to run your function on it) and 2. a zip containing all the images I want to train on
# #These two files need to be present in the root folder of the project that has the notebook. In jupyter I guess you could use the toolbar to directly upload these files without using a command.
# #I created a zip file with 1000 images called 'mini_train.zip' and will be using that from here on.

# In[ ]:


x = constructY('train.csv')


# In[ ]:


imageCol = ['ImageId_ClassId']
imageNames = pd.read_csv('train.csv', names=imageCol)


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


# In[ ]:


print(defects.count(0))  # No defects in 5902 images


# In[ ]:


print(defects.count(1)) # Defects present in 6666 images


# In[ ]:


print(len(images))
images = list(dict.fromkeys(images))
print(len(images))


# In[ ]:


import os

directory = os.fsencode('./')
miniTrain_images_names = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        miniTrain_images_names.append(filename)
        continue
    else:
        continue

#creating a list with the names of the 100 images uploaded to the directory


# In[ ]:


miniTrain_images_names.sort()
miniTrain_images_names = miniTrain_images_names[0:100]
print(len(miniTrain_images_names))


# In[ ]:


import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


# In[ ]:


import cv2
import numpy as np

for name in miniTrain_images_names:
    in_img = name
    if 1:
        img = cv2.imread(in_img)
          #get size
        height, width, channels = img.shape
          #print (in_img,height, width, channels)
          # Create a black image
        x = height if height > width else width
        y = height if height > width else width
        square= np.zeros((x,y,3), np.uint8)
        square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
        cv2.imwrite(name,square)
        cv2.waitKey(0)

#using opencv to convert all the rectangular images to squares (by just padding black pixels around)


# In[ ]:


i = 0
plt.figure(figsize=(10,10))
img = imread('000a4bcdd.jpg')
plt.imshow(img)

#to check how the squared image looks (size 1600 x 1600)


# In[ ]:


mini_train_imgs = []
i = 0
for name in miniTrain_images_names:
    image_path = str(name)
    img = imread(image_path, as_grey = True)
    img = img/255.0
    img = img.astype('float32')
    mini_train_imgs.append(img)

print(img.shape)

#since images cant directly be fed into the cnn, we convert the image into a matrix of it's pixel values. So each image is represented by a unique 1600 x 1600 matrix
#We convert the image to grayscale to make things simpler. divide by 255 to normalize the pixel values. i.e now each value in matrix lies in [0,1]


# In[ ]:


print(len(mini_train_imgs)) #sanity-check


# In[ ]:


mini_train_data = np.array(mini_train_imgs)

#Converting the image pixel matrices into a numpy array


# In[ ]:


print((mini_train_data[0][800])) #sanity-check


# In[ ]:


mini_labels_list = []
for imgname in miniTrain_images_names:
  for i in range(len(images)):
    if images[i] == imgname:
      mini_labels_list.append(defects[i])

#building the list of labels - i.e for image 'mini_train_data[k]' the value of the label would be 'mini_labels_list[k]'


# In[ ]:


print(len(mini_labels_list))


# In[ ]:


mini_train_labels = np.array(mini_labels_list)

#making the list of labels into a numpy array


# In[ ]:


i = 0
plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(mini_train_data[i], cmap='gray')
plt.subplot(222), plt.imshow(mini_train_data[i+26], cmap='gray')
plt.subplot(223), plt.imshow(mini_train_data[i+50], cmap='gray')
plt.subplot(224), plt.imshow(mini_train_data[i+75], cmap='gray')

#more sample images (squared)


# In[ ]:


train_x, val_x, train_y, val_y = train_test_split(mini_train_data, mini_train_labels, test_size = 0.2)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

#making a train set and a validation set (80:20 ratio) => 80 training images, 20 validation images


# In[ ]:


train_x = train_x.reshape(80, 1, 1600, 1600)
train_x  = torch.from_numpy(train_x)

train_y = train_y.astype(int);
train_y = torch.from_numpy(train_y)

train_x.shape, train_y.shape

#We reshape the train set into the dimensions required - 80 images, 1600 x 1600 matrix (for each image)
#convert train images into a tensor for pytorch using torch.from_numpy


# In[ ]:


val_x = val_x.reshape(20, 1, 1600, 1600)
val_x  = torch.from_numpy(val_x)

val_y = val_y.astype(int);
val_y = torch.from_numpy(val_y)

#convert test images into a tensor for pytorch using torch.from_numpy


# In[ ]:


import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class SimpleCNN(Module):
  #Defining the cnn model   
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 2 convolutional layers, 1 fully connected
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 397 * 397, 2) 
        
    def forward(self, x):
        # after each conv layer, to add a max pool layer
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x)) # flattening
        x = self.fc1(x) #relu
        return x


    def num_flat_features(self, x): 
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[ ]:


model = SimpleCNN()
print(model)


# In[ ]:


optimizer = Adam(model.parameters(), lr=1e-3) #optimizer
criterion = CrossEntropyLoss() 
# checking if GPU is available
#if torch.cuda.is_available():
#    model = model.cuda()
#    criterion = criterion.cuda()

#training function

def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    #if torch.cuda.is_available():
    #    x_train = x_train.cuda()
    #    y_train = y_train.cuda()
    #    x_val = x_val.cuda()
    #    y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)


# In[ ]:


n_epochs = 5 # setting number of epochs
# empty list to store training losses
train_losses = [] 
# empty list to store validation losses
val_losses = []
for epoch in range(n_epochs):
    train(epoch)


# In[ ]:


train_losses 


# In[ ]:


val_losses


# In[ ]:


from sklearn.metrics import accuracy_score
from tqdm import tqdm

with torch.no_grad():
    output = model(train_x)
    
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
accuracy_score(train_y, predictions) 


# In[ ]:


plt.plot(train_losses)
plt.ylabel('Training losses')
plt.xlabel('No. of iterations')
plt.title('Learning rate 1e-3')
plt.show()


# In[ ]:


plt.plot(val_losses)
plt.ylabel('Validation losses')
plt.xlabel('No. of iterations')
plt.title('Learning rate 1e-3')
plt.show()

