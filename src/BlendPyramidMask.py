import cv2
import numpy as np
import math
from LaplacianPyramid import *
from GaussianPyramid import *
from utilsHw import *

class BlendPyramidMask(LaplacianPyramid):
    def __mix(self, img1, img2, mask):
        assert img1.shape == img2.shape == mask.shape
        img1 = img1.astype(float)
        img2 = img2.astype(float)
        mask = (mask.astype(float) / 255).astype(float)
        return (mask * img1 + (1 - mask) * img2)


    def __init__(self, img1, img2, mask, levels):
        self.image = self.__mix(img1, img2, mask).astype(int)
        debug("mask", mask.astype('uint8'))
        #debug("blendit!", self.image.astype(np.uint8))
        self.levels = levels
        self.__img1_laplacian = LaplacianPyramid(img1, levels)
        self.__img2_laplacian = LaplacianPyramid(img2, levels)
        self._LaplacianPyramid__img_arr = []
        self.__mask_pyramid = GaussianPyramid(mask.astype(int), levels)
        #debug("down", self.down(0))
        for i in range(levels):
          self._LaplacianPyramid__img_arr.append(self.__mix(self.__img1_laplacian.access(i),
            self.__img2_laplacian.access(i),
            self.__mask_pyramid.access(i)))



