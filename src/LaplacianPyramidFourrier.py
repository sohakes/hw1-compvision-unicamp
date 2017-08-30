import cv2
import numpy as np
import math
import time
from GaussianPyramidFourrier import *
from utilsHw import *

class LaplacianPyramidFourrier:
    def up(self, level): #downsample
        #by definition, from the paper
        if level == self.levels - 1:
            return self.__gaussian_pyramid.access(level).copy()
        gauss_img = self.__gaussian_pyramid.down(level + 1).copy()
        lapl_img = self.__gaussian_pyramid.access(level) - gauss_img

        return lapl_img

    def down(self, img): #upsample
        #first x direction

        width, height = img.shape
        padw, padh = int(width/2), int(height/2)
        return np.pad(img, ((padw, padw), (padh, padh)), 'constant') * 4

    def recover_original(self): #upsample
        img = self.__img_arr[self.levels - 1]
        for i in range(self.levels - 2, -1, -1):
            img = self.down(img) + self.__img_arr[i]

        return img

    #public
    def access(self, level):
        return self.__img_arr[level]

    def __init__(self, image, levels):
        self.image = image
        self.levels = levels
        self.__gaussian_pyramid = GaussianPyramidFourrier(image, levels)
        self.__img_arr = []
        for i in range(levels):
          self.__img_arr.append(self.up(i))

