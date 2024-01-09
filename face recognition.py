#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import face_recognition as fr


# In[2]:


get_ipython().system('pip install face-recognition')


# In[3]:


get_ipython().run_cell_magic('cmd', '', 'pip install "C:\\\\dlib-19.22.99-cp39-cp39-win_amd64.whl"')


# In[4]:


get_ipython().run_cell_magic('cmd', '', 'where python')


# In[5]:


import cv2
import numpy as np
import face_recognition as fr


# In[6]:


video_capture = cv2.VideoCapture (0)
image = fr.load_image_file( 'harris.jpg')
image_face_encoding = fr.face_encodings (image)[0]
known_face_encodings = [image_face_encoding]
known_face_names = ["Harris"]
while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    fc_locations = fr.face_locations (rgb_frame)
    fc_encodings = fr.face_encodings (rgb_frame, fc_locations)
    for (top, right, bottom, left), face_encoding in zip(fc_locations, face_encodings):
        mathces = fr.comapre_faces (known_face_encodings, face_encoding)
        name="unknown"
        fc_distances = fr.face_distances (known_face_encodings, face_encoding)
        match_index = np.argmin(fc_distances)
        I
        if matches [match_index]:
            name = known_face_names[match_index]
        cv2. rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), CV2. FILLED)
        font= cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText (frame, name, (left +6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Simplilearn face detection system', frame)
    if cv2.waitkey (1) & 0xFF == ord( 'q'):
        break
video_capture.release()
cv2.destroyAllWindows()


# In[1]:


import cv2
import face_recognition as fr
import numpy as np

video_capture = cv2.VideoCapture(0)

image = fr.load_image_file('harris.jpg')
image_face_encoding = fr.face_encodings(image)[0]
known_face_encodings = [image_face_encoding]
known_face_names = ["Harris"]

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    fc_locations = fr.face_locations(rgb_frame)
    fc_encodings = fr.face_encodings(rgb_frame, fc_locations)

    for (top, right, bottom, left), face_encoding in zip(fc_locations, fc_encodings):
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        fc_distances = fr.face_distance(known_face_encodings, face_encoding)
        match_index = np.argmin(fc_distances)

        if matches[match_index]:
            name = known_face_names[match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Simplilearn face detection system', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




