# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:54:55 2022

@author: Vikanksh
"""
# for now()
import datetime
 
# for timezone()
import pytz
 
# using now() to get current time
current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

video_file = str(current_time.year) + '_' + str(current_time.month) + '_' + str(current_time.day) \
    + '_' +  str(current_time.hour) + '_' + str(current_time.minute) + '_' + str(current_time.second)

import cv2 as cv

capture = cv.VideoCapture(0) #to open Camera

#accessing pretrained model
pretrained_model = cv.CascadeClassifier("haarcascade_frontalface_default.xml") 
# pretrained_model = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

vid_cod = cv.VideoWriter_fourcc(*'XVID')
output = cv.VideoWriter("videos/"+video_file+'.mp4', vid_cod, 20.0, (640,480))

i=0
while True:
    boolean, frame = capture.read()
    # https://stackoverflow.com/questions/38563079/opencv-python-cv2-cv-cap-prop-pos-frames-error
    # frame_no = capture.get(1) #retrieves the current frame number
    
    i = i+1
    frame_no = i
    if boolean == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        coordinate_list = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) 
        
        
        # drawing rectangle in frame
        for (x,y,w,h) in coordinate_list:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
        text_on_video = 'Frame: ' + str(int(frame_no))
            
        # describe the type of font
        # to be used.
        font = cv.FONT_HERSHEY_SIMPLEX
      
        # Use putText() method for
        # inserting text on video
        cv.putText(frame, 
                    text_on_video, 
                    (50, 50), 
                    font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv.LINE_4)
                
        # Display detected face
        cv.imshow("Live Face Detection", frame)
        
        output.write(frame)
        
        # condition to break out of while loop by pressing 'x' on keyboard
        if cv.waitKey(20) == ord('x'):
            break
        
capture.release()
output.release()
cv.destroyAllWindows()
