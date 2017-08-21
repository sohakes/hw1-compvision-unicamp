import cv2
import numpy as np
import math
from LaplacianPyramid import *
from utilsHw import *

class BlendPyramid(LaplacianPyramid):
    def __mix(self, img1, img2):
        assert img1.shape == img2.shape
        img1 = img1.astype(int)
        img2 = img2.astype(int)
        middle = math.floor(img1.shape[1]/2)
        img1half = img1[:, :middle]
        middlecolumn = (img1[:, middle] + img2[:, middle]) / 2
        img2half = img2[:, middle + 1:]
        #print("shapes:", img1half.shape, img2half.shape, middlecolumn.shape)
        halfs = np.concatenate((img1half, img2half), axis=1).astype(int)
        return np.insert(halfs, middle, middlecolumn, axis=1)


    def __init__(self, img1, img2, levels):
        self.image = self.__mix(img1, img2).astype(int)
        #debug("blendit!", self.image.astype(np.uint8))
        self.levels = levels
        self.__img1_laplacian = LaplacianPyramid(img1, levels)
        self.__img2_laplacian = LaplacianPyramid(img2, levels)
        self._LaplacianPyramid__img_arr = []
        #debug("down", self.down(0))
        for i in range(levels):
          self._LaplacianPyramid__img_arr.append(self.__mix(self.__img1_laplacian.access(i), self.__img2_laplacian.access(i)))
          #debug("blend", self._LaplacianPyramid__img_arr[-1])

        img = self.recover_original()
        img = img + np.min(img)
        debug("original!", img.astype(np.uint8))
