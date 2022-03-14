#!/usr/bin/env python
# coding: utf-8

# # Train Face Recognition Model

# #### Import Libraries

# In[6]:


import numpy as np
import cv2
import os
# import the face recognition file we just created that has the classifier in it
import Face_Recognition as fr


# In[ ]:





# #### Give path to the image you want to test the model with

# In[7]:


test_img = cv2.imread(r'dataset\test_images\image0000.jpg')


# In[8]:


print(test_img)


# In[9]:


# Feed the model with the test img
faces_detected, gray_img = fr.face_detection(test_img)
print('Face Detected: ', faces_detected)


# #### Train the model 

# In[ ]:


faces, faceID = fr.labels_for_training_data(r'dataset\images')
face_recognizer = fr.train_classifier(faces, faceID) # get the classifier and feed it faces with the id so it can correctly recognize
face_recognizer.save('trainingData.yml') # save the model to this location 


# In[ ]:


#name = {0:'Thobela', 1: "Bobo"} # zero is the label, so if there are more people write it as a long dictionary
name = {0:'Thobela'}


# In[ ]:


for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y+w, x: x+h]  # roi means reason of interest, it should be y+w
    label, confidence = face_recognizer.predict(roi_gray) 
    print('Confidence : ', confidence) # confidence is how confident the model is with the recognition
    print('label : ', label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.place_text(test_img, predicted_name, x, y)


# In[ ]:


resized_img = cv2.resize(test_img, (1000, 700))


# In[ ]:


cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




