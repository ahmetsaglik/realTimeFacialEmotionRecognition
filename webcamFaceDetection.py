import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Trained model added
model = model_from_json(open("fer.json","r").read()) 

# Weight values added
model.load_weights('faceModel.h5')

video = cv2.VideoCapture(0) # webcam image capture

# Cascade file used for face detection has been added.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# An infinite loop was created to see the incoming image.
while(1):
    _,frame = video.read()

    frame = cv2.flip(frame,1)

    # the received frame has been converted to gray format
    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    
    faces = face_cascade.detectMultiScale(grayFrame,1.3,3)

    # Detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        roiGray = grayFrame[y:y+h,x:x+w]
        roiGray = cv2.resize(roiGray,(48,48))
        imgPixels = image.img_to_array(roiGray)
        imgPixels = np.expand_dims(imgPixels,axis=0)
        imgPixels /= 255

        # Estimates are made using the model.
        predictions = model.predict(imgPixels)
        print(predictions)

        maxIndex = np.argmax(predictions)

        emotions = ('sinirli' , 'igrenmis' , 'korkmus' , 'mutlu' , 'uzgun' , 'saskin' , 'notr')
        predictedEmotion = emotions[maxIndex]

        cv2.putText(frame,predictedEmotion,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

    resizedImage = cv2.resize(frame,(1000,700))
    cv2.imshow("Kamera",resizedImage)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



video.release()
cv2.destroyAllWindows()
