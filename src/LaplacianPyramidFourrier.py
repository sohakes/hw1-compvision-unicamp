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
        #debug("lasplimg1",  inverse_fourier_transform(lapl_img))
        #debug("lasplimg2",  (inverse_fourier_transform(self.__gaussian_pyramid.access(level)) - inverse_fourier_transform(gauss_img)).astype('uint8'))
        #width, height = gauss_img.shape
        #gauss_img[0:int(width/4)+1] = 0
        #gauss_img[int(width - width/4)+1:0] = 0
        #gauss_img[0:int(height/4)+1] = 0
        #gauss_img[int(height - height/4)+1:0] = 0
        #minl = np.min(lapl_img)
        #if minl < 0:
        #    lapl_img = lapl_img - minl
        return lapl_img
        #return fourrier_transform(inverse_fourier_transform(self.__gaussian_pyramid.access(level)) - inverse_fourier_transform(gauss_img))

    def down(self, img): #upsample
        #first x direction

        width, height = img.shape
        padw, padh = int(width/2), int(height/2)
        return np.pad(img, ((padw, padw), (padh, padh)), 'constant') * 4

    def recover_original(self): #upsample
        img = self.__img_arr[self.levels - 1]
        for i in range(self.levels - 2, -1, -1):
            img = self.down(img) + self.__img_arr[i]

        #img = img / (self.levels - 1)
        return img

    #public
    def access(self, level):
        return self.__img_arr[level]

    def __init__(self, image, levels):
        self.image = image
        self.levels = levels
        self.__gaussian_pyramid = GaussianPyramidFourrier(image, levels)
        self.__img_arr = []
        #debug("down", self.down(0))
        for i in range(levels):
          self.__img_arr.append(self.up(i))
          #debug("up", self.__img_arr[-1])

        #img = self.recover_original()
        #debug("original!", img.astype(np.uint8))
