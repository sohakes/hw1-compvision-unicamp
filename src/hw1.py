import cv2
import numpy as np
import math
import time
from BlendPyramid import *
from LaplacianPyramid import *
from GaussianPyramid import *
from utilsHw import *

################  HW1  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################

def question_convolution(img):
    conv = convolution(img, np.array([[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]]))
    debug('conv',conv)
    cv2.imwrite('output/p1-2-1-0.png', conv)
    convocv = convolution_opencv(img, np.array([[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]]))
    debug('convocv',convocv)
    cv2.imwrite('output/p1-2-1-1.png', convocv)

    gaussian_mask = create_gaussian_mask(5, 3)
    gaussian_img = convolution(img, gaussian_mask)
    debug('gaussian_img',gaussian_img)
    cv2.imwrite('output/p1-2-2-0.png', gaussian_img)

def question_gaussianpyramid():
    # colocar um exemplo soh com a gaussian pyramid
    return  'colocar cidugi aqyu'

def question_laplacianpyramid():
    # colocar um exemplo soh com a laplacian pyramid    
    return  'colocar cidugi aqyu'

def question_spacialblending():
    #gaussian_pyramid = GaussianPyramid(img, 5)
    img = cv2.imread('input/p1-1-4.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('input/p1-1-3.png', cv2.IMREAD_GRAYSCALE)
    img = np.pad(img, ((0, 1), (0, 1)), 'edge')
    img2 = np.pad(img2, ((0, 1), (0, 1)), 'edge')
    laplacian_pyramid = BlendPyramid(img, img2, 6)
    #Agora precisa fazer a parte com mask!
    #não pular!

def question_fourierspace():
    return  'colocar cidugi aqyu'

def question_frequencyblending():
    return  'colocar cidugi aqyu'

def main():
    img = cv2.imread('input/p1-1-1.png', cv2.IMREAD_GRAYSCALE)
    question_convolution(img)
    question_gaussianpyramid()
    question_laplacianpyramid()
    question_spacialblending()
    question_spacialblending()
    question_fourierspace()
    question_frequencyblending()

if __name__ == '__main__':
   main()

