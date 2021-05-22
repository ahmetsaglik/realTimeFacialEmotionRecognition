#coding:utf-8-*-

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

numClasses = 7
batchSize = 256
epochs = 50

# Files are pulled from the dataset in rows.
with open("fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

numInstances = lines.size
print(numInstances)  # Returns the number of rows.


xTrain,yTrain,xTest,yTest = [],[],[],[]

# Lines labeled as "Training" with for loop returning as many lines as yTrain and xTrain lines
# labeled as "PublicTest" are assigned to variables yTest and xTest.
for i in range(1,numInstances):
    emotion,img,usage = lines[i].split(",")
    val = img.split(" ")
    pixels = np.array(val,"float32")
    emotion = keras.utils.to_categorical(emotion,numClasses)

    if "Training" in usage:
        yTrain.append(emotion)
        xTrain.append(pixels)
    elif "PublicTest" in usage:
        yTest.append(emotion)
        xTest.append(pixels)

print(yTrain)
exit()
# Variables are converted to float data type.
xTrain = np.array(xTrain,"float32")
yTrain = np.array(yTrain,"float32")
xTest = np.array(xTest,"float32")
yTest = np.array(yTest,"float32")

xTrain /= 255
xTest /= 255

# Incoming pixel values are transformed into images by reshaping (48 * 48).
xTrain = xTrain.reshape(xTrain.shape[0],48,48,1)
xTrain = xTrain.astype("float32")
xTest = xTest.reshape(xTest.shape[0],48,48,1)
xTest = xTest.astype("float32")

print(xTrain.shape[0]," Train images") # Returns the number of images reserved for training.
print(xTest.shape[0]," Test images")  # Returns the number of images reserved for testing.


## Creating a Deep Learning Model
model = Sequential()

model.add(Conv2D(64,kernel_size=(3,3),kernel_initializer="he_normal",input_shape=(48,48,1),activation="relu"))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(128,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(256,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(512,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

# With Flatten, data is transformed into vectors so that it can be added to neurons.
model.add(Flatten())

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(7,activation="softmax"))

# Summary of the model
model.summary()

# The loss function, optimizer and performance metric of the model were determined.
model.compile(loss="categorical_crossentropy" , optimizer="Adam" , metrics=["accuracy"])

# Checkpointer ensures that a better result is recorded during training.
checkpointer = ModelCheckpoint(filepath="faceModel.h5",verbose=1,save_best_only=True)

# The data are placed on the prepared model and the training starts by determining the number of epochs and the size of the cluster.
hist = model.fit(np.array(xTrain),np.array(yTrain),
                 batch_size=batchSize,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(np.array(xTest),np.array(yTest)),
                 shuffle=True,
                 callbacks=[checkpointer])


# The completed training is saved as a json file so that it can be used later.
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)


# Showing the obtained values with graphics
plt.figure(figsize=(14,3))
plt.subplot(1,2,1)
plt.suptitle("Eğitim",fontsize=10)
plt.ylabel("loss",fontsize=16)
plt.plot(hist.history["loss"],color="b",label="Training loss")
plt.plot(hist.history["val_loss"],color="r",label="Validation loss")
plt.legend(loc="upper right")

plt.subplot(1,2,2)
plt.ylabel("Accuracy",fontsize=16)
plt.plot(hist.history["accuracy"],color="b",label="Training Accuracy")
plt.plot(hist.history["val_accuracy"],color="r",label="Validation Accuracy")
plt.legend(loc="lower right")
plt.show()


































