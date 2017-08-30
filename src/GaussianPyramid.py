import cv2
import numpy as np
import math
import time
from utilsHw import *

#IMPORTANT:
#there is an assumption that the image has width and height defined as 2^n + 1
#The assumption was made in the paper and I've maintained it
class GaussianPyramid:
    def up(self, level): #downsample
        img = convolution(self.__img_arr[level], self.__gaussian_mask)
        return img[::2, ::2]

        #private
    def down(self, level): #upsample
        #first x direction
        img = self.__img_arr[level]
        upimg = img.astype(int)
        for i in range(img.shape[1] - 1, 0, -1):
            result_column = (upimg[:, i] + upimg[:, i - 1]) / 2
            upimg = np.insert(upimg, i, result_column, axis=1)

        for i in range(img.shape[0] - 1, 0, -1):
            result_row = (upimg[i, :] + upimg[i - 1, :]) / 2
            upimg = np.insert(upimg, i, result_row, axis=0)
        return upimg.astype(np.uint8)

    #public
    def access(self, level):
        return self.__img_arr[level]

    def __init__(self, image, levels):
        self.__gaussian_mask = create_gaussian_mask(5, 3)
        self.image = image
        self.levels = levels
        self.__img_arr = [image]
        for i in range(levels - 1):
          self.__img_arr.append(self.up(i))

