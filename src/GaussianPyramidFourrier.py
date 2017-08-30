import cv2
import numpy as np
import math
import time
from utilsHw import *

#IMPORTANT:
#there is an assumption that the image has width and height defined as 2^n + 1
#The assumption was made in the paper and I've maintained it
class GaussianPyramidFourrier:
    def up(self, level): #downsample
        img = self.__img_arr[level].copy()
        width, height = img.shape
        assert width == height #in this case
        #apply blur as we see here
        #https://classroom.udacity.com/courses/ud810/lessons/3490398569/concepts/35009385480923
        imgabs = np.abs(img) * np.abs(fourrier_transform(create_gaussian_mask_fourrier(31, 3, width)))

        #if we don't do this, and simply multiply, the quadrants get mixed for some weird reason
        img = imgabs * np.exp(1j*np.angle(img))
        resimg = img[int(width/4):int(width - width/4)+1, int(height/4):int(height - height/4)+1]/4
        return resimg

        #private
    def down(self, level):
        img = self.__img_arr[level]
        width, height = img.shape
        padw, padh = int(width/2), int(height/2)
        mean = np.mean(np.abs(img))
        #simple pad it
        fimg = np.pad(img, ((padw, padw), (padh, padh)), 'constant')
        return fimg * 4 #scale it

    #public
    def access(self, level):
        return self.__img_arr[level]

    def __init__(self, image, levels):
        self.image = image
        self.levels = levels
        self.__img_arr = [image]
        for i in range(levels - 1):
          self.__img_arr.append(self.up(i))

