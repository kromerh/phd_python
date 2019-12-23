import scipy.io as sio
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

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

    def preprocess_for_KERAS_reshaping(self, framesize, dataset):
        """
        Preprocessing for the dataset to be used in KERAS.
        INPUT:
            - dataset: numpy array with shape (framesize, framesize, #examples). Should be
                        after the grayscaling step!
            - framesize: number that depicts the size of the frame, i.e. 32x32
        OUTPUT:
            - dataset that is still a numpy array. Shape is (#examples, framesize, framesize, 1)
        """
        dataset = np.rollaxis(dataset,2)


        dataset = dataset.reshape(-1, framesize, framesize, 1)

#         print(f'Dataset reshaped to: {dataset.shape}')

        return dataset


    def preprocess_for_KERAS_labels(self, labels_dataset):
        """
        Preprocessing for the labels of dataset to be used in KERAS. Converts 10 to 0, and reshapes.
        INPUT:
            - labels_dataset: numpy array with shape (#examples,1).
        OUTPUT:
            - labels_dataset that is still a numpy array. Shape is (#examples,). 10 is replaced with 0
        """
        labels_dataset = labels_dataset[:,0]
        labels_dataset[labels_dataset==10] = 0

        return labels_dataset


    def model_definition(self):
        """
        Builds the model for the digit detection.
        Taken from https://nbviewer.jupyter.org/github/dyckia/SVHN-CNN/blob/master/SVHN.ipynb.
        """

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
        return model