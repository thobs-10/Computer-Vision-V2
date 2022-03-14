#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys


# In[2]:


cpt = 0 # count


# In[3]:


vidStream = cv2.VideoCapture(0)
# while the camera is on, take images
while True:
    ret, frame = vidStream.read()
    # show the image
    cv2.imshow('Test Frame', frame)
    
    cv2.imwrite(r"dataset\images\0\image%04i.jpg" %cpt, frame)
    # after taking pictures every second, increase the number
    cpt +=1
    # if i press the q it shoudld stop taking pictures
    if cv2.waitKey(10) == ord('q'):
        break


# In[ ]:




