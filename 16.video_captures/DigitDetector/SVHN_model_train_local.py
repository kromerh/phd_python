import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SVHNDataset():

    def load_dataset(self, path_train, path_test):
        """
        Loads the .mat file from the SVHN Dataset (train and test) indicated at location path. Returns it as numpy array,
        """
        train_dataset = sio.loadmat(path_train)
        test_dataset = sio.loadmat(path_test)

        train_data, train_labels = train_dataset['X'], train_dataset['y']
        test_data, test_labels = test_dataset['X'], test_dataset['y']

        print( 'Train data:', train_data.shape,', Train labels:', train_labels.shape )
        print( 'Test data:', test_data.shape,', Test labels:', test_labels.shape )

        return train_data, train_labels, test_data, test_labels

    def convert_to_gray(self, data):
        """
        Converts all the images in the dataset into gray scale. Returns the dataset with grayscale entries.
        """

        r, g, b = data[:,:,0,:], data[:,:,1,:], data[:,:,2,:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        data[:,:,0,:] = gray
        data = data[:,:,0,:]

        return data

svhn = SVHNDataset()
path_train  = '/Users/hkromer/02_PhD/02_Data/12.dcr/Stanford_housenumbers/train_32x32.mat'
path_test  = '/Users/hkromer/02_PhD/02_Data/12.dcr/Stanford_housenumbers/test_32x32.mat'

train_data, train_labels, test_data, test_labels = svhn.load_dataset(path_train, path_test)
# convert to grayscale
train_data = svhn.convert_to_gray(train_data)
test_data = svhn.convert_to_gray(test_data)
print('After conversion to grayscale: ')
print(f'Train data: {train_data.shape}, labels: {train_labels.shape}')
print(f'Test data: {test_data.shape}, labels: {test_labels.shape}')

X_train = np.rollaxis(train_data,2)
X_test = np.rollaxis(test_data,2)

X_train = X_train.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)

print(f'Train data: {X_train.shape}')
print(f'Test data: {X_test.shape}')

y_train = train_labels[:,0]
y_test = test_labels[:,0]

print(f'Train labels: {y_train.shape}')
print(f'Test  labels: {y_test.shape}')


y_train[y_train==10] = 0
y_test[y_test==10] = 0



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


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
    Dense(10, activation='softmax')
])


# get a summary of our built model
print(model.summary())


# define the optimizer, loss function and metrics for the network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# start training
history = model.fit(X_train, y_train, epochs=10)
model.save('/Users/hkromer/02_PhD/02_Data/12.dcr/Stanford_housenumbers/2019-19-21.KERAS_model.h5')
# loss, acc = model.evaluate(X_test, y_test)
# print("Model accuracy on test data is: {:6.3f}%".format(100 * acc))

