import cv2
import numpy as np
import math
import time
from BlendPyramid import *
from BlendPyramidMask import *
from LaplacianPyramid import *
from GaussianPyramid import *
from utilsHw import *

################  HW1  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################

def question_convolution(img):
    #filter_conv = [[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]]
    #filter_conv = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.2,0.1,0.1,0.2,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.2]]
    filter_conv = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1]]
    filter_conv = filter_conv/np.sum(filter_conv)
    conv = convolution(img, np.array(filter_conv))
    debug('conv',conv)
    cv2.imwrite('output/p1-2-1-0.png', conv)
    convocv = convolution_opencv(img, np.array(filter_conv))
    debug('convocv',convocv)
    cv2.imwrite('output/p1-2-1-1.png', convocv)

    gaussian_mask = create_gaussian_mask(5, 3)
    gaussian_img = convolution(img, gaussian_mask)
    debug('gaussian_img',gaussian_img)
    cv2.imwrite('output/p1-2-2-0.png', gaussian_img)

def question_gaussianpyramid(img):
    gaussian_pyramid = GaussianPyramid(img, 5)
    access = gaussian_pyramid.access(4)
    cv2.imwrite('output/gp-4.png', access)
    access = gaussian_pyramid.access(3)
    cv2.imwrite('output/gp-3.png', access)
    access = gaussian_pyramid.access(2)
    cv2.imwrite('output/gp-2.png', access)
    access = gaussian_pyramid.access(1)
    cv2.imwrite('output/gp-1.png', access)

def question_laplacianpyramid(img):
    laplacian_pyramid = LaplacianPyramid(img, 5)
    access = laplacian_pyramid.access(4)
    cv2.imwrite('output/lp-4.png', access)
    access = laplacian_pyramid.access(3)
    cv2.imwrite('output/lp-3.png', access)
    access = laplacian_pyramid.access(2)
    cv2.imwrite('output/lp-2.png', access)
    access = laplacian_pyramid.access(1)
    cv2.imwrite('output/lp-1.png', access)

def question_spacialblending():
    img = cv2.imread('input/p1-1-4.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('input/p1-1-3.png', cv2.IMREAD_GRAYSCALE)
    img = np.pad(img, ((0, 1), (0, 1)), 'edge')
    img2 = np.pad(img2, ((0, 1), (0, 1)), 'edge')
    #blend_pyramid = BlendPyramid(img, img2, 6)
    print(create_circular_mask(20,20,10,10,5))
    blend_pyramid_mask = BlendPyramidMask(img, img2,
     create_circular_mask(img.shape[0], img.shape[1], 220, 220, 200), 6)

def question_fourierspace(img):
    f = fourrier_transform(img)
    m = magnitude(f)
    # save magnitude
    cv2.imwrite('output/magnitude.png', 20*np.log(m))
    debug("magnitude", 20*np.log(m).astype('uint8'))
    p = phase(f)
    # save phase
    cv2.imwrite('output/phase.png', (p))
    debug("phase", (p).astype('uint8'))
    ift = inverse_fourier_transform(f, 25.0, 100.0, 0.0, 0.0)
    debug("inverse", ift)
    ift = inverse_fourier_transform(f, 100.0, 99.0, 0.0, 0.0)
    debug("inverse", ift)
    ift = inverse_fourier_transform(f, 100.0, 100.0, 25.0, 0.0)
    debug("inverse", ift)
    ift = inverse_fourier_transform(f, 100.0, 100.0, 0.0, 99.0)
    debug("inverse", ift)
    ift = inverse_fourier_transform(f, -1, 100.0, 0.0, 0.0)
    debug("inverse", ift)
    ift = inverse_fourier_transform(f, 100.0, -1, 0.0, 0.0)
    debug("inverse", ift)
    ift = inverse_fourier_transform(f, 100.0, 100.0, -1, 0.0)
    debug("inverse", ift)
    ift = inverse_fourier_transform(f, 100.0, 100.0, 0.0, -1)
    debug("inverse", ift)
    cv2.imwrite('output/inverse.png', ift.astype('uint8'))
    debug("inverse", ift)


def question_frequencyblending():
    img = cv2.imread('input/p1-1-4.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('input/p1-1-3.png', cv2.IMREAD_GRAYSCALE)
    img = np.pad(img, ((0, 1), (0, 1)), 'edge')
    img2 = np.pad(img2, ((0, 1), (0, 1)), 'edge')
    rimg = frequency_blend(img, img2,
     create_circular_mask(img.shape[0], img.shape[1], 220, 220, 200))
    debug("rimg", rimg)
    


def main():
    img = cv2.imread('input/p1-1-1.png', cv2.IMREAD_GRAYSCALE)
    #question_convolution(img)
    #question_gaussianpyramid(img)
    #question_laplacianpyramid(img)
    #question_spacialblending()
    #question_spacialblending()
    #question_fourierspace(img)
    #question_spacialblending()
    question_frequencyblending()

if __name__ == '__main__':
   main()
