from numpy.random import randint

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Display the contents of the specified directory
print(os.listdir("B:/AI_project/DataSets/"))

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization, MaxPooling2D, Flatten, Conv2D, Dense
)

SIZE = 256 # Image size for resizing


def readFiles(directoryPath, imagesRF, imagesDT, labels):
    # Iterate through each subdirectory in the specified path
    for directory_path_Train in glob.glob(directoryPath):
        label = directory_path_Train.split("\\")[-1]
        print(label)
        # Iterate through each image in the subdirectory
        for img_path in glob.glob(os.path.join(directory_path_Train, "*.jpg")):
            print(img_path)

            # Read and resize the image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            imgRF = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgDT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imagesRF.append(imgRF)
            imagesDT.append(imgDT)
            labels.append(label)

# Lists to store training and testing data
train_imagesDT = []
train_imagesRF = []
train_labels = []

test_imagesDT = []
test_imagesRF = []
test_labels = []

# Read training data
readFiles('B:/AI_project/DataSets/Training/*', train_imagesRF, train_imagesDT, train_labels)
# Read testing data
readFiles('B:/AI_project/DataSets/Testing/*', test_imagesRF, test_imagesDT, test_labels)

# Convert lists to NumPy arrays
train_imagesDT = np.array(train_imagesDT)
train_imagesRF = np.array(train_imagesRF)
train_labels = np.array(train_labels)

test_imagesDT = np.array(test_imagesDT)
test_imagesRF = np.array(test_imagesRF)
test_labels = np.array(test_labels)

################ Label encoding using sklearn's preprocessing #################
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

################ Split data for training and testing #################

x_trainDT, x_trainRF, y_train, x_testDT, x_testRF, y_test = train_imagesDT, train_imagesRF, train_labels_encoded, \
                                                            test_imagesDT, test_imagesRF, test_labels_encoded
# Normalize pixel values to be between 0 and 1
x_trainRF, x_testRF = x_trainRF / 255.0, x_testRF / 255.0
x_trainDT, x_testDT = x_trainDT / 255.0, x_testDT / 255.0

################## Flatten 2D arrays for training and testing data ##############

n_samples_train, nx_train, ny_train = x_trainDT.shape
d2_x_dataset_train = x_trainDT.reshape((n_samples_train, nx_train * ny_train))

n_samples_test, nx_test, ny_test = x_testDT.shape
d2_x_dataset_test = x_testDT.reshape((n_samples_test, nx_test * ny_test))

################ One-hot encode the target labels ################

from keras.utils import to_categorical
 
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

################# Decision Tree Classifier ################

from sklearn.tree import DecisionTreeClassifier

x_DT_model = DecisionTreeClassifier(criterion='entropy', max_depth=50, random_state=42)
x_DT_model.fit(d2_x_dataset_train, y_train)
prediction_DT = x_DT_model.predict(d2_x_dataset_test)
prediction_DT = le.inverse_transform(prediction_DT)

################# Evaluate Decision Tree model ################

from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from numpy import mean
from numpy import std

print('Accuracy for Decision Tree  = ', metrics.accuracy_score(test_labels, prediction_DT))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_DT)
print('Confusion matrix for Decision Tree\n\n', cm)

################# VGG16 model for feature extraction ################

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

feature_extractor = VGG_model.predict(x_trainRF)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_RF = features

################# Random Forest Classifier ################

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std

RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

RF_model.fit(X_for_RF, y_train)

x_test_feature = VGG_model.predict(x_testRF)
x_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)

prediction_RF = RF_model.predict(x_test_features)
prediction_RF = le.inverse_transform(prediction_RF)

################# Evaluate Random Forest model ###############

from sklearn import metrics

print('Accuracy for Random forest = ', metrics.accuracy_score(test_labels, prediction_RF))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)

print('Confusion matrix for Random Forest\n\n', cm)

################ GUI using Tkinter ################

from tkinter import *
from tkinter import messagebox
from numpy.random import randint
from PIL import Image, ImageTk
import matplotlib

matplotlib.use("TkAgg")

app = Tk()

app.title("welcome to Project")
app.geometry("350x350+120+120")

from sklearn import metrics

# Function to display Decision Tree accuracy in a message box
def DecisionTreeAccuracy():
    messagebox.showinfo("Accuracy for Decision Tree. \n", str(metrics.accuracy_score(test_labels, prediction_DT)))

# Button for Decision Tree accuracy
btn1 = Button(app, text="DecisionTree Accuracy ", width=80, height=3, bg="#a09f9f", fg="blue", borderwidth=0,
              command=DecisionTreeAccuracy)
btn1.pack()

# Function to display Random Forest accuracy in a message box
def RandomForestAccuracy():
    messagebox.showinfo("Accuracy for Random Forests.\n", str(metrics.accuracy_score(test_labels, prediction_RF)))

# Button for Random Forest accuracy
btn2 = Button(app, text="Random Forest Accuracy", width=80, height=3, bg="#a09f9f", fg="blue", borderwidth=0,
              command=RandomForestAccuracy)
btn2.pack()

# Function to display a Random Forest prediction for a random test image
def RandomForest():
    app4 = Tk()
    app4.geometry("600x600")

    n = randint(0, len(x_testRF))
    img = x_testRF[n]

    input_img = np.expand_dims(img, axis=0)
    input_img_features = VGG_model.predict(input_img)
    input_img_features = input_img_features.reshape(input_img_features.shape[0], -1)

    prediction_RF = RF_model.predict(input_img_features)[0]
    prediction_RF = le.inverse_transform([prediction_RF])
    image = Image.fromarray(test_imagesRF[n])
    img = ImageTk.PhotoImage(image=image, master=app4)
    image = Label(app4, text=("The prediction for this image is: ", prediction_RF), image=img)
    image.config(compound='bottom')
    print("The actual label for this image is: ", test_labels[n])
    image.pack()
    app4.mainloop()

# Button for displaying a Random Forest prediction for a random test image
btn3 = Button(app, text="RF prediction", width=80, height=3, bg="#a09f9f", fg="red", borderwidth=0, command=RandomForest)
btn3.pack()

# Function to display a Decision Tree prediction for a random test image
def DecisionTree():
    app5 = Tk()
    app5.geometry("600x600")

    n = randint(0, len(x_testRF))
    img = d2_x_dataset_test[n]
    input_img = np.expand_dims(img, axis=0)
    input_img_features = x_DT_model.predict(input_img)

    prediction_DT = x_DT_model.predict(input_img)[0]

    prediction_DT = le.inverse_transform([prediction_DT])
    image = Image.fromarray(test_imagesDT[n])
    img = ImageTk.PhotoImage(image=image, master=app5)
    image = Label(app5, text=("The prediction for this image is: ", prediction_DT), image=img)
    image.config(compound='bottom')
    image.pack()
    app5.mainloop()

# Button for displaying a Decision Tree prediction for a random test image
btn4 = Button(app, text="DT prediction", width=80, height=3, bg="#a09f9f", fg="red", borderwidth=0, command=DecisionTree)
btn4.pack()


####################### Heat Map For Random Forest ############################

def HeatMapRF():
    cm = confusion_matrix(test_labels, prediction_RF)
    sns.heatmap(cm, annot=True)
    plt.show()


btn5 = Button(app, text="Heat Map For Random Forest", width=80, height=3, bg="#a09f9f", fg="black", borderwidth=0,
              command=HeatMapRF)
btn5.pack()

####################### Heat Map For Decision Tree ############################

def HeatMapDT():
    cm = confusion_matrix(test_labels, prediction_DT)
    sns.heatmap(cm, annot=True)
    plt.show()


btn6 = Button(app, text="Heat Map For Decision Tree", width=80, height=3, bg="#a09f9f", fg="black", borderwidth=0,
              command=HeatMapDT)
btn6.pack()


# Run the Tkinter main loop
app.mainloop()