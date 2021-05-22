# realTimeFacialEmotionRecognition
Real time facial emotion recognition using Python and Keras

The training was carried out using the "fer2013.csv" data set, which can be accessed free of charge from the Kaggle platform.
The trained model was then saved as "fer.json" and "faceModel.h5" for effective use.
Real-time face detection system to use the registered model is included in the "webcamFaceDetection.py" file.
Face detection was done using Haar-like features.

<h2> CONTENT </h2>

faceModel.h5 -> Model weight file <br>
fer.json -> Saved model <br>
fer2013.csv -> Dataset from Kaggle <br>
FaceRec.py -> Model training codes <br>
haarcascade_frontalface_default.xml -> Haar-like file used for face detection <br>
webcamFaceDetection.py -> Written codes for real-time face detection and emotion prediction <br>
