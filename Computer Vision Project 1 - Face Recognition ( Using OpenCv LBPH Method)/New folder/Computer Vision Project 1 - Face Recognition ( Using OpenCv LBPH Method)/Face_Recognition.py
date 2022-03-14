#!/usr/bin/env python
# coding: utf-8

# # Facial Recognition

# How LBPH operation works:
# 
# - The method gets the training image and uses a sliding window operation to get all the pixels and try to create an intermediate image that looks similar or is a replica of the original by using the features it got from the training image/data. What essentially happens is,  the model gets an image and from that image it tries to get every pixel of the image and gets the pixels of the features. Through that it sets a threshold for pixels, where it says pixels under the specified threshold are not important and pixels that are greater than the threshold are important.
# 
# - The pixels which are regarded as important are labeled 1 and the non important one are labeled 0. The model cannot take all the features from the image and flag them as important because by doing that it will make the model hard to perform facial recognition. So after the threshold step all the pixels are converted to binary values.
# 

# ### Import libraries

# In[1]:


import numpy as np
import pandas as pd
import cv2 # to convert any image to pixel form
import os 


# #### Detect the face

# In[2]:


#get_ipython().system('pip install opencv-contrib-python')


# In[3]:


#!pip install opencv-python


# In[4]:


def face_detection(input_img):
    '''detect an image from an input image or from a video'''
    # convert the image to grayscale
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # to detect the face from the gray image
    # NB: Download haarcascade_frontalface_alt it is an xml file and place it anywhere you want so you can access that file
    face_haar = cv2.CascadeClassifier(r"C:\Users\Cash Crusaders\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")
    # from that omage we gonna extract the face
    # scale factor scaales down the image so that it can detect the face properly from the whole image
    faces = face_haar.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 3)
    return faces, gray_img
    


# In[5]:


# test the method by inputing the a random picture that has a face in it.
face_detection(cv2.imread(r'dataset\test_images\image0000.jpg'))


# #### Creating labels for training data

# In[6]:


# NB: FIRST CREATE IMAGES TO TRAIN THE MODEL, CREATE DATASET FROM WEBCAM


# In[7]:


def labels_for_training_data(directory):
    '''create labels from the pictures taken using the webcam'''
    faces = [] 
    faceID = []
    
    for path, subdirnames, filenames in os.walk(directory): # fiilenames are the  images in the 0 folder
        for filename in filenames:
            if filename.startswith("."): # if anything starts with a dot then it is an error
                print("Skipping the system file")
                continue
            id = os.path.basename(path) # get the image path
            img_path = os.path.join(path, filename) # join it witth the filename
            print("img_path", img_path)
            print("id: ", id)
            input_img = cv2.imread(img_path) # read the joined image of path and filename
            if input_img is None: # if the image directory is empoty
                print("Not loaded properly")
                continue
            # if the files are there, label them
            faces_rect, gray_img = face_detection(input_img) # make rectangle on this face from the gray image, on that rect angle do something
            (x, y, w, h) = faces_rect[0] # make the rectangle, because the my images are in zero, for someone else their images will be in 1 or another number
            roi_gray = gray_img[y:y+w, x:x+h] # decllare the sides of the box 
            faces.append(roi_gray) 
            faceID.append(int(id))
    return faces, faceID
            
            


# #### Training the Classifier

# In[8]:


def train_classifier(faces, faceID):
    '''train the model to recognize the faces'''
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID)) # the face id of the person is associated with the image of that person
    return face_recognizer


# #### Drawing a Rectangle on the face 

# In[9]:


def draw_rect(input_img, face):
    '''it will draw the rectangle around the detected face on the image/video'''
    (x, y, w, h) = face
    cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)


# #### Place text on the detect image/face

# In[10]:


def place_text(input_img, label_name, x, y):
    '''it will try to place a name of the detected person above the rectangle that surrounds the face'''
    cv2.putText(input_img, label_name, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)


# In[ ]:





# In[ ]:




