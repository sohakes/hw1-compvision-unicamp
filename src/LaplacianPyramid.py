import cv2
import numpy as np
import math
import time
from GaussianPyramid import *
from utilsHw import *

class LaplacianPyramid:
    def up(self, level): #downsample
        #by definition, from the paper
        if level == self.levels - 1:
            return self.__gaussian_pyramid.access(level)
        gauss_img = self.__gaussian_pyramid.down(level + 1).astype(int)
        lapl_img = self.__gaussian_pyramid.access(level).astype(int) - gauss_img

        return lapl_img

    def down(self, img): #upsample
        #first x direction
        upimg = img.astype(int)
        for i in range(img.shape[1] - 1, 0, -1):
            result_column = (upimg[:, i] + upimg[:, i - 1]) / 2
            upimg = np.insert(upimg, i, result_column, axis=1)

        for i in range(img.shape[0] - 1, 0, -1):
            result_row = (upimg[i, :] + upimg[i - 1, :]) / 2
            upimg = np.insert(upimg, i, result_row, axis=0)
        return upimg


    def recover_original(self): #upsample
        img = self.__img_arr[self.levels - 1]
        for i in range(self.levels - 2, -1, -1):
            img = self.down(img) + self.__img_arr[i]

        return np.clip(img, 0, 255).astype(np.uint8)

    #summation property
    def recover_originalbkp(self): #upsample
        sumimg = []
        for i in range(self.levels):
            img = self.__img_arr[i]
            for j in range(i):
                img = self.down(img)
            if sumimg == []:
                sumimg = img.astype(float)
            else:
                sumimg = sumimg + img.astype(float)
        return (sumimg / self.levels).astype(np.uint8)

    #public
    def access(self, level):
        return self.__img_arr[level]

    def __init__(self, image, levels):
        self.image = image.astype(int)
        self.levels = levels
        self.__gaussian_pyramid = GaussianPyramid(image, levels)
        self.__img_arr = []
        for i in range(levels):
          self.__img_arr.append(self.up(i))

        img = self.recover_original()
        debug("original!", img.astype(np.uint8))
