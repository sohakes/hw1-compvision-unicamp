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
    # Filter for tests
    filter_conv = [[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]]
    #filter_conv = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.2,0.1,0.1,0.2,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.2]]
    #filter_conv = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1]]
    
    # Normalization
    filter_conv = filter_conv/np.sum(filter_conv)

    # Split channels
    b,g,r = cv2.split(img)  
    convb = convolution(b, np.array(filter_conv))
    convg = convolution(g, np.array(filter_conv))
    convr = convolution(r, np.array(filter_conv))
    # Merge  channels    
    conv = cv2.merge((b, g, r))
    debug('conv',conv)
    cv2.imwrite('output/p1-2-1-0.png', conv)

    # Split channels opencv    
    convb = convolution_opencv(b, np.array(filter_conv))
    convg = convolution_opencv(g, np.array(filter_conv))
    convr = convolution_opencv(r, np.array(filter_conv))
    # Merge channels opencv    
    convocv =  cv2.merge((b, g, r))
    debug('convocv',convocv)
    cv2.imwrite('output/p1-2-1-1.png', convocv)

    gaussian_mask = create_gaussian_mask(5, 3)
    gaussian_img_r = convolution(r, gaussian_mask)
    gaussian_img_g = convolution(g, gaussian_mask)
    gaussian_img_b = convolution(b, gaussian_mask)
    gaussian_img = cv2.merge((gaussian_img_b, gaussian_img_g, gaussian_img_r))
    debug('gaussian_img',gaussian_img)
    cv2.imwrite('output/p1-2-2-0.png', gaussian_img)

def question_gaussianpyramid(img):
    b,g,r = cv2.split(img) 
    gaussian_pyramid_r = GaussianPyramid(r, 5)
    gaussian_pyramid_g = GaussianPyramid(g, 5)
    gaussian_pyramid_b = GaussianPyramid(b, 5)

    # Merge channels and write access    
    access = merge_color( gaussian_pyramid_b, gaussian_pyramid_g, gaussian_pyramid_r,4)   
    cv2.imwrite('output/gp-4.png', access)

    # Merge channels and write access
    access = merge_color( gaussian_pyramid_b, gaussian_pyramid_g, gaussian_pyramid_r,3)   
    cv2.imwrite('output/gp-3.png', access)

    # Merge channels and write access
    access = merge_color( gaussian_pyramid_b, gaussian_pyramid_g, gaussian_pyramid_r,2)   
    cv2.imwrite('output/gp-2.png', access)

    # Merge channels and write access
    access = merge_color( gaussian_pyramid_b, gaussian_pyramid_g, gaussian_pyramid_r,1)   
    cv2.imwrite('output/gp-1.png', access)

    # Merge channels and write access
    access = merge_color( gaussian_pyramid_b, gaussian_pyramid_g, gaussian_pyramid_r,0)   
    cv2.imwrite('output/gp-0.png', access)

def question_laplacianpyramid(img):
    b,g,r = cv2.split(img) 
    laplacian_pyramid_r = LaplacianPyramid(r, 5)
    laplacian_pyramid_g = LaplacianPyramid(g, 5)
    laplacian_pyramid_b = LaplacianPyramid(b, 5)

    # Merge channels and write access
    access = merge_color( laplacian_pyramid_b, laplacian_pyramid_g, laplacian_pyramid_r,4) 
    cv2.imwrite('output/lp-4.png', access)

    # Merge channels and write access
    access = merge_color( laplacian_pyramid_b, laplacian_pyramid_g, laplacian_pyramid_r,3) 
    cv2.imwrite('output/lp-3.png', access)

    # Merge channels and write access
    access = merge_color( laplacian_pyramid_b, laplacian_pyramid_g, laplacian_pyramid_r,2) 
    cv2.imwrite('output/lp-2.png', access)

    # Merge channels and write access    
    access = merge_color( laplacian_pyramid_b, laplacian_pyramid_g, laplacian_pyramid_r,1) 
    cv2.imwrite('output/lp-1.png', access)

    # Merge channels and write access
    access = merge_color( laplacian_pyramid_b, laplacian_pyramid_g, laplacian_pyramid_r,0) 
    cv2.imwrite('output/lp-0.png', access)

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
    return  'colocar cidugi aqyu'

def main():
    img = cv2.imread('input/p1-1-1.png')
  
    question_convolution(img)
    question_gaussianpyramid(img)
    question_laplacianpyramid(img)
    #question_spacialblending()
    #question_spacialblending()
    #question_fourierspace(img)
    #question_spacialblending()
    #question_frequencyblending()

if __name__ == '__main__':
   main()
