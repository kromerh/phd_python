from sklearn.model_selection import train_test_split
import os
import scipy.io as sio
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import re
import numpy as np
from SVHNDataset import SVHNDataset
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # otherwise there will be an error

# list of the image files in the training set
folder_images = '/Users/hkromer/02_PhD/02_Data/12.dcr/Stanford_housenumbers/train/'

files_images = os.listdir(folder_images)
files_images = [f"{folder_images}{f}" for f in files_images if f.endswith('png')]

# load images
images = []
#4 corners
framesize = 32
# can be made a lot more efficient using numpy arrays
images_corners = np.empty([4*len(files_images),framesize,framesize]) # it will be at most 4 times the size of the original images

jj = 0 # index in images_corners
for ii, file in enumerate(files_images):
    img = cv2.imread(file, 0) #grayscale
    # convert to grayscale
    width = img.shape[1]
    height = img.shape[0]
    # take only images that are significantly larger so that there will not be too much of the digit in the image
    if (width > 50) & (height > 50):
        images.append(img)
        # extrat the 4 corners in each image

        # left positions of the corner
        c1 = img[0:framesize,0:framesize].copy() # top left
        c2 = img[0:framesize,width-framesize:].copy() # top right
        c3 = img[height-framesize:,0:framesize].copy() # bottom left
        c4 = img[height-framesize:,width-framesize:].copy() # bottom right
        images_corners[jj] = c1
        jj += 1
        images_corners[jj] = c2
        jj += 1
        images_corners[jj] = c3
        jj += 1
        images_corners[jj] = c4
        jj += 1

# not all the images satisfied the width and height criteria, adjust the images_corner array
s = np.sum(np.sum(images_corners,axis=1),axis=1)
s_limit = s[s>0].shape[0]

images_corners = images_corners[0:s_limit,:,:]

print(f'Loaded images in the corner, shape of dataset: {images_corners.shape}')

svhn = SVHNDataset()


data_negative = images_corners # this will be y=0 class

# preprocessing for keras, because it wants image numbers, frame, frame, channels
data_negative = svhn.preprocess_for_KERAS_reshaping(32, data_negative)

print('After preprocessing reshaping: ')
print(f'Negative dataset: {data_negative.shape}')


# load the dataset with the house numbers from SVHN
# these do all contain numbers, so that will be the positive dataset, y=1

path_train  = '/Users/hkromer/02_PhD/02_Data/12.dcr/Stanford_housenumbers/train_32x32.mat'
path_test  = '/Users/hkromer/02_PhD/02_Data/12.dcr/Stanford_housenumbers/test_32x32.mat'

train_data, train_labels, test_data, test_labels = svhn.load_dataset(path_train, path_test)
# convert to grayscale
train_data = svhn.convert_to_gray(train_data)
test_data = svhn.convert_to_gray(test_data)
print(' ')
print('After conversion to grayscale: ')
print(f'Original SVHN train data: {train_data.shape}, labels: {train_labels.shape}')
print(f'Original SVHN test data: {test_data.shape}, labels: {test_labels.shape}')

X_train = svhn.preprocess_for_KERAS_reshaping(32, train_data)
X_test = svhn.preprocess_for_KERAS_reshaping(32, test_data)
print(' ')
print('After preprocessing reshaping: ')
print(f'Original SVHN train data: {X_train.shape}')
print(f'Original SVHN test data: {X_test.shape}')

# combine the original test and train dataset into one
data_positive = np.concatenate((X_train, X_test))

print(' ')
print(f'Positive dataset: {data_positive.shape}')

# shuffle the positive and the negative dataset
np.random.shuffle(data_positive)
np.random.shuffle(data_negative)

# make the two classes equal
data_positive = data_positive[0:data_negative.shape[0],:,:,:]
print(' ')
print('Made two classes equal in size')
print(f'Positive dataset: {data_positive.shape}')
print(f'Negative dataset: {data_negative.shape}')

labels_negative = np.zeros(data_negative.shape[0]) # y = 0
labels_positive = np.ones(data_positive.shape[0]) # y = 0

# join the two classes
X = np.concatenate((data_positive,data_negative))
y = np.concatenate((labels_positive,labels_negative))

# shuffle
X, y = shuffle(X, y, random_state=0)

print(' ')
print('Shuffled and joined the classes')
print(f'Features X: {X.shape}, labels y: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax') # either 1 or 0
])

# define the optimizer, loss function and metrics for the network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# start training
history = model.fit(X_train, y_train, epochs=10)

model.save('/Users/hkromer/02_PhD/02_Data/12.dcr/Stanford_housenumbers/2019-19-23.KERAS_model_DigitDetector.h5')
