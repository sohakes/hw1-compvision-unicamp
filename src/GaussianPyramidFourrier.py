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
        #we can just crop the image, and it will automatically blur before that since
        #by cropping you are removing the higher frequencies. After that you need to scale
        #the magnitudes
        #mean = np.mean(np.abs(img))
        #img = img * create_gaussian_mask2(img.shape[0],10) * mean
        #resimg = img[int(width/4):int(width - width/4)+1, int(height/4):int(height - height/4)+1]/4
        
        # http://fourier.eng.hmc.edu/e161/lectures/fourier/node15.html
        
        
        #kind of works
        """
        width = resimg.shape[0]
        for i in range(width):
            for j in range(width):
                u = i - int(width/2)
                v = j - int(width/2)
                d = int(width) * 25
                #d = np.sqrt(u**2 + v**2)
               # if d == 0:
                #    d = 1
                m = np.exp(-3*(u**2 + v**2)/d)
                resimg[i, j] = resimg[i, j] * m
        """
        #https://classroom.udacity.com/courses/ud810/lessons/3490398569/concepts/35009385480923
        #debug('img0', inverse_fourier_transform(img))
        imgabs = np.abs(img) * np.abs(fourrier_transform(create_gaussian_mask_fourrier(31, 3, width)))
        img = imgabs * np.exp(1j*np.angle(img))
        #debug('img1', inverse_fourier_transform(img))
        #debug('img filtro true true', create_gaussian_mask3(width, 0.001) * 255)
        #debug('img filtro true', create_gaussian_mask_fourrier2(31, 3, width))
        #debug('img filtro', np.abs(fourrier_transform(create_gaussian_mask_fourrier2(5, 3, width))))
        resimg = img[int(width/4):int(width - width/4)+1, int(height/4):int(height - height/4)+1]/4
        #debug('resimg', inverse_fourier_transform(resimg))
        return resimg

        #private
    def down(self, level): #upsample
        #first x direction
        img = self.__img_arr[level]
        width, height = img.shape
        padw, padh = int(width/2), int(height/2)
        mean = np.mean(np.abs(img))
        fimg = np.pad(img, ((padw, padw), (padh, padh)), 'constant')
        return fimg * 4

    #public
    def access(self, level):
        return self.__img_arr[level]

    def __init__(self, image, levels):
        #self.__gaussian_mask = create_gaussian_mask(5, 3)
        self.image = image
        self.levels = levels
        self.__img_arr = [image]
        #debug("down", self.down(0))
        for i in range(levels - 1):
          self.__img_arr.append(self.up(i))
          print("shape level", i, self.__img_arr[-1].shape)
          #debug("up", self.__img_arr[-1])

