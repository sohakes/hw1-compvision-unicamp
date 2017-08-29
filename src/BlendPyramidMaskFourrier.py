import cv2
import numpy as np
import math
from LaplacianPyramidFourrier import *
from GaussianPyramidFourrier import *
from utilsHw import *

class BlendPyramidMaskFourrier(LaplacianPyramidFourrier):
    def convolve(self, img1, img2):
        def convolve_point(img1, img2, x, y):
            s = 0
            for i in range(x + 1):
                for j in range(y + 1):
                    s += img1[i, j] * img2[x-i, y-j]
            return s
        #imgr = img1.copy()
        #for i2 in range(img1.shape[0]):
        #    for j2 in range(img1.shape[1]):
        #        imgr[i2, j2] = convolve_point(img1, img2, i2, j2)
        #print(imgr)
        return signal.convolve2d(img1, img2)

    def __mix(self, img1, img2, mask):
        assert img1.shape == img2.shape == mask.shape
        img1 = inverse_fourier_transform(img1).astype(float)
        img2 = inverse_fourier_transform(img2).astype(float)
        mask = (inverse_fourier_transform(mask).astype(float) / 255).astype(float)
        return fourrier_transform((mask * img1 + (1 - mask) * img2))


    def __init__(self, img1, img2, mask, levels):
        self.image = self.__mix(img1, img2, mask)
        #debug("blendit!", self.image.astype(np.uint8))
        self.levels = levels
        self.__img1_laplacian = LaplacianPyramidFourrier(img1, levels)
        self.__img2_laplacian = LaplacianPyramidFourrier(img2, levels)
        self._LaplacianPyramidFourrier__img_arr = []
        self.__mask_pyramid = GaussianPyramidFourrier(mask, levels)
        #debug("down", self.down(0))
        for i in range(levels):
            self._LaplacianPyramidFourrier__img_arr.append(self.__mix(self.__img1_laplacian.access(i),
                self.__img2_laplacian.access(i),
                self.__mask_pyramid.access(i)))
            debug("blendit!", inverse_fourier_transform(self._LaplacianPyramidFourrier__img_arr[-1]).astype(np.uint8))




