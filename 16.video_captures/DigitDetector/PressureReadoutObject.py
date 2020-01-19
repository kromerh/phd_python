import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PressureReadoutObject():

    def __init__(self, credentials_file):
        # load the credentials
        credentials = pd.read_csv(credentials_file, header=0)
        pw = credentials['password'].values[0]
        self.password = pw
        self.host = 'fastdd01' # camera feed of the camera pointing on the pressure readout
        self.user = 'admin' # connect to the camera

    def read_camera(self):
        """
        Connects to the camera feed, reads it and returns an array of the pixel values.
        """
        url = f'http://admin:{self.password}@fastdd01/video.cgi'
        cap = cv2.VideoCapture(url)
        r, f = cap.read()
        if r == True:
            return f
        else:
            print('ERROR! Could not read camera...')
            return -1

    def read_test_image(self,path):
        """
        Reads a pre-saved test image at location path (csv file) and returns it.
        """
        img = pd.read_csv(path, index_col=0).values

        return img

    def convert_rgb2gray(self, rgb):

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    def scale_image(self, img, scale_perc):
        """
        Takes an image img (numpy array) and returns a scaled (in percent) down version of the image (as np array.)
        """
#         scale_perc = 40 # percent of original size
        width = int(img.shape[1] * scale_perc / 100)
        height = int(img.shape[0] * scale_perc / 100)
        dim = (width, height)

        scaled_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        return scaled_img