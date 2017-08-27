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

NUMBER_FILE = -1

# define number file for convolution
def numFile():
    global NUMBER_FILE
    NUMBER_FILE = NUMBER_FILE + 1
    return NUMBER_FILE 
    

def question_convolution(img, filter_conv):
    # Normalization
    filter_conv = filter_conv/np.sum(filter_conv)

    # Split channels
    b,g,r = cv2.split(img)  
    convb = convolution(b, np.array(filter_conv))
    convg = convolution(g, np.array(filter_conv))
    convr = convolution(r, np.array(filter_conv))
    # Merge  channels    
    conv = cv2.merge((convb, convg, convr))
    debug('conv',conv)
    cv2.imwrite('output/p1-2-1-'+ str(numFile()) +'.png', conv)

    # Split channels opencv    
    convb = convolution_opencv(b, np.array(filter_conv))
    convg = convolution_opencv(g, np.array(filter_conv))
    convr = convolution_opencv(r, np.array(filter_conv))
    # Merge channels opencv    
    convocv =  cv2.merge((convb, convg, convr))
    debug('convocv',convocv)

    cv2.imwrite('output/p1-2-1-'+ str(numFile())+'.png', convocv)

    gaussian_mask = create_gaussian_mask(5, 3)
    gaussian_img_r = convolution(r, gaussian_mask)
    gaussian_img_g = convolution(g, gaussian_mask)
    gaussian_img_b = convolution(b, gaussian_mask)
    gaussian_img = cv2.merge((gaussian_img_b, gaussian_img_g, gaussian_img_r))
    debug('gaussian_img',gaussian_img)

    cv2.imwrite('output/p1-2-1-'+ str(numFile()) +'.png', gaussian_img)


def question_gaussianpyramid(img, level):
    b,g,r = cv2.split(img) 
    gaussian_pyramid_r = GaussianPyramid(r, level)
    gaussian_pyramid_g = GaussianPyramid(g, level)
    gaussian_pyramid_b = GaussianPyramid(b, level)

    # Merge channels and write access    
    for i in range(0,level):
        access = merge_color( gaussian_pyramid_b, gaussian_pyramid_g, gaussian_pyramid_r,i)   
        cv2.imwrite('output/p1-2-2-'+ str(i) +'.png', access)


def question_laplacianpyramid(img,level):
    b,g,r = cv2.split(img) 
    laplacian_pyramid_r = LaplacianPyramid(r, 5)
    laplacian_pyramid_g = LaplacianPyramid(g, 5)
    laplacian_pyramid_b = LaplacianPyramid(b, 5)

    # Merge channels and write access
    for i in range(0,level):
        access = merge_color( laplacian_pyramid_b, laplacian_pyramid_g, laplacian_pyramid_r,i) 
        cv2.imwrite('output/p1-3-'+ str(i) +'.png', access)


def question_spacialblending(namefile, namefile2, hasMask, namefile_final):
    img = cv2.imread('input/' + str(namefile))
    img2 = cv2.imread('input/'+ str(namefile2))
    
    b,g,r = cv2.split(img)
    b2,g2,r2 = cv2.split(img2)  
    
    b = np.pad(b, ((0, 1), (0, 1)), 'edge')
    b2 = np.pad(b2, ((0, 1), (0, 1)), 'edge')
    g = np.pad(g, ((0, 1), (0, 1)), 'edge')
    g2 = np.pad(g2, ((0, 1), (0, 1)), 'edge')    
    r = np.pad(r, ((0, 1), (0, 1)), 'edge')
    r2 = np.pad(r2, ((0, 1), (0, 1)), 'edge')

    if (hasMask):
        mask_r = create_circular_mask(r.shape[0], r.shape[1], 220, 220, 200)
        mask_g = create_circular_mask(g.shape[0], g.shape[1], 220, 220, 200)
        mask_b = create_circular_mask(b.shape[0], b.shape[1], 220, 220, 200)
        pyramid_r = BlendPyramidMask(r, r2, mask_r,5)
        pyramid_g = BlendPyramidMask(g, g2, mask_g,5)
        pyramid_b = BlendPyramidMask(b, b2, mask_b,5)
    else:
        pyramid_r = BlendPyramid(r, r2, 5)
        pyramid_g = BlendPyramid(g, g2, 5)
        pyramid_b = BlendPyramid(b, b2, 5)
    
    img_r = pyramid_r.recover_original()
    img_g = pyramid_g.recover_original() 
    img_b = pyramid_b.recover_original()
 
    img = cv2.merge((img_b, img_g, img_r)) 
    cv2.imwrite('output/' + str(namefile_final),  img.astype(np.uint8))    

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
    img = cv2.imread('input/p1-1-1.png')
  
    # Test : filter 3 x 3   
    filter_conv = [[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]]
    question_convolution(img, filter_conv)   

    # Test filter 7 x 7
    filter_conv = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    ,[0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1]]
    question_convolution(img, filter_conv)   

    # Test filter 15 x 15
    filter_conv = [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    ,[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1]
    ,[0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    ,[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1]
    ,[0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    ,[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1]
    ,[0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    ,[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1]
    ,[0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    ,[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1]
    ,[0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.1]]
    question_convolution(img, filter_conv) 
    
    # Test: level 5
    question_gaussianpyramid(img, 5)

    # Test: level 5  
    question_laplacianpyramid(img, 5)

    # Test : Without mask
    question_spacialblending('p1-1-3.png','p1-1-4.png',False,'p1-4-0.png')
    question_spacialblending('p1-1-5.png','p1-1-6.png',False,'p1-4-1.png')
    
    # Test: With  mask
    question_spacialblending('p1-1-7.png','p1-1-8.png',True,'p1-4-2.png')
    #question_spacialblending('p1-1-5.png','p1-1-6.png',True,'p1-4-3.png')

    #question_fourierspace(img)
    #question_spacialblending()
    #question_frequencyblending()

if __name__ == '__main__':
   main()
