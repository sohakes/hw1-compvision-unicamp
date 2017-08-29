import cv2
import numpy as np
import math
import time
from BlendPyramid import *
from BlendPyramidMask import *
from BlendPyramidMaskFourrier import *
from LaplacianPyramid import *
from LaplacianPyramidFourrier import *
from GaussianPyramid import *
from GaussianPyramidFourrier import *
from utilsHw import *

################  HW1  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################

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
        cv2.imwrite('output/p1-2-3-'+ str(i) +'.png', access)

def question_spacialblending(filename, filename2, hasMask, filename_final, mask_type, maskfile):
    img = cv2.imread('input/' + str(filename))
    img2 = cv2.imread('input/'+ str(filename2))
    
    b,g,r = cv2.split(img)
    b2,g2,r2 = cv2.split(img2)  
    
    b = np.pad(b, ((0, 1), (0, 1)), 'edge')
    b2 = np.pad(b2, ((0, 1), (0, 1)), 'edge')
    g = np.pad(g, ((0, 1), (0, 1)), 'edge')
    g2 = np.pad(g2, ((0, 1), (0, 1)), 'edge')    
    r = np.pad(r, ((0, 1), (0, 1)), 'edge')
    r2 = np.pad(r2, ((0, 1), (0, 1)), 'edge')

    if (hasMask):
        mask_r,mask_g, mask_b = defineMask(r,g,b, mask_type , maskfile)        
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
    cv2.imwrite('output/' + str(filename_final),  img.astype(np.uint8))    

def question_fourierspace(img, m):
    b,g,r = cv2.split(img)
    f_r = fourrier_transform(r)
    f_g = fourrier_transform(g)
    f_b = fourrier_transform(b)
    
    m_r = magnitude(f_r)
    m_g = magnitude(f_g)
    m_b = magnitude(f_b)

    cv2.imwrite('output/p1-5-0.png', (20*np.log(m_r)).astype('uint8'))
    cv2.imwrite('output/p1-5-1.png', (20*np.log(m_g)).astype('uint8'))
    cv2.imwrite('output/p1-5-2.png', (20*np.log(m_b)).astype('uint8'))
    
    p_r = phase(f_r)
    p_g = phase(f_g)
    p_b = phase(f_b)
    
    cv2.imwrite('output/p1-5-3.png', p_r.astype('uint8'))
    cv2.imwrite('output/p1-5-4.png', p_g.astype('uint8'))
    cv2.imwrite('output/p1-5-5.png', p_b.astype('uint8'))

    r = len(m)

    for i in range(0,r):
        ift_r = inverse_fourier_transform(f_r,m[i][0],m[i][1],m[i][2],m[i][3])
        ift_b = inverse_fourier_transform(f_b,m[i][0],m[i][1],m[i][2],m[i][3])
        ift_g = inverse_fourier_transform(f_g,m[i][0],m[i][1],m[i][2],m[i][3])
        ift = cv2.merge((ift_b, ift_g, ift_r)) 
        cv2.imwrite('output/p1-3-1-'+ str(i+6) +'.png', ift)    


def question_frequencyblending(filename, filename2, filename_final, mask_type, maskfile):
    img = cv2.imread('input/' + str(filename))
    img2 = cv2.imread('input/'+ str(filename2))
    
    b,g,r = cv2.split(img)
    b2,g2,r2 = cv2.split(img2)  
    
    b = np.pad(b, ((0, 1), (0, 1)), 'edge')
    b2 = np.pad(b2, ((0, 1), (0, 1)), 'edge')
    g = np.pad(g, ((0, 1), (0, 1)), 'edge')
    g2 = np.pad(g2, ((0, 1), (0, 1)), 'edge')    
    r = np.pad(r, ((0, 1), (0, 1)), 'edge')
    r2 = np.pad(r2, ((0, 1), (0, 1)), 'edge')
    
    mask_r,mask_g, mask_b = defineMask(r,g,b, mask_type , maskfile)  
    fb_r = frequency_blend(r, r2 , mask_r)
    fb_g = frequency_blend(g, g2 , mask_g)
    fb_b = frequency_blend(b, b2 , mask_b)
    fb = cv2.merge((fb_b,fb_g,fb_r))
    cv2.imwrite('output/'+ filename_final +'.png', fb.astype(np.uint8)) 


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
    question_spacialblending('p1-1-3.png','p1-1-4.png',False,'p1-2-4-0.png', None, None)
    question_spacialblending('p1-1-5.png','p1-1-6.png',False,'p1-2-4-1.png', None, None)
    
    # Test: With  mask
    question_spacialblending('p1-1-7.png','p1-1-8.png',True,'p1-2-4-2.png', 'circle', None)
    question_spacialblending('p1-1-10.png','p1-1-11.png',True,'p1-2-4-3.png', None, 'p1-1-9.png')

    # Test: Each row has:
    #       col 1: percentage_phase_up , 
    #       col 2: percentage_magnitude_up , 
    #       col 3: percentage_phase_down ,
    #       col 4: percentage_magnitude_down

    vals = [-1, 25.0, 50.0, 75.0, 100.0]
    
    m = [[[x if i == j else 100.0 for j in range(4)] for x in vals] for i in range(4)]
    m = sum(m, [])
    question_fourierspace(img,m)
    """
    # Test: With mask
    question_frequencyblending('p1-1-4.png','p1-1-3.png','p1-6-0.png', 'circle', None)
    

    img = cv2.imread('input/p1-1-1.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('input/p1-1-10.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('input/p1-1-11.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('input/p1-1-9.png', cv2.IMREAD_GRAYSCALE)
    img1 = np.pad(img1, ((0, 1), (0, 1)), 'edge')
    img2 = np.pad(img2, ((0, 1), (0, 1)), 'edge')
    mask = np.pad(mask, ((0, 1), (0, 1)), 'edge')
    imgf = fourrier_transform(img1)
    img1f = fourrier_transform(img1)
    img2f = fourrier_transform(img2)
    maskf = fourrier_transform(mask)
    
    p = GaussianPyramidFourrier(img1f, 5)
    
    pback = inverse_fourier_transform(p.access(1))
    p5 = GaussianPyramid(img1, 5)
    debug("gaussian fourrier", pback)    
    debug("gaussian orig", p5.access(1).astype('uint8'))
    p3 = LaplacianPyramid(img1, 5)
    
    p2 = LaplacianPyramidFourrier(imgf, 5)
    p4 = BlendPyramidMaskFourrier(img1f,img2f,maskf,5)
    p6 = BlendPyramidMask(img1,img2,mask,5)
    p2back = inverse_fourier_transform(p2.access(2))
    debug("p2back", p2back)
    debug("laplac original", inverse_fourier_transform(p2.recover_original()))
    debug("blend fourrier", inverse_fourier_transform(p4.recover_original()/2))
    debug("blend original", p6.recover_original())
    debug("p3back", p3.access(2).astype('uint8'))
    """

if __name__ == '__main__':
   main()
