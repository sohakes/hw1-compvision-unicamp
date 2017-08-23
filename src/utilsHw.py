import cv2
import numpy as np
import math
import time

DEBUG = True

def debug(name,img):
    if DEBUG == False:
        return

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_gaussian_mask(size, sigma):
    assert size % 2 == 1, "mask should have odd size"
    def pixel_val(x, y):
        #return np.exp(-(X.^2 + Y.^2) / (2*sigma*sigma));
        return (1.0/(2 * math.pi * sigma ** 2)) * math.e**(-(x**2 + y**2)/(2*sigma**2))

    halfsize = math.floor(size / 2)

    mask = np.array([[pixel_val(i, j) for i in range(-halfsize, halfsize + 1)] for j in range(-halfsize, halfsize + 1)])
    msum = np.sum(mask)

    return mask / msum

#got it from https://stackoverflow.com/questions/5478351/python-time-measure-function
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap

@timing
def convolution(inputimg, mask):
    middle = (math.floor(mask.shape[0] / 2), math.floor(mask.shape[1] / 2))
    pad = max(middle)
    #we think that edge padding makes the image behave less unexpectedly
    #at the borders
    out = inputimg.copy()
    img = np.pad(inputimg, pad, 'edge')
    for i in range(pad, inputimg.shape[0] - pad):
        for j in range(pad, inputimg.shape[1] - pad):
            out[i - pad, j - pad] = np.sum(img[i - middle[0]:i + middle[0] + 1, j - middle[1]:j + middle[1] + 1] * mask)

    return out

@timing
def convolution_opencv(inputimg, mask):
    out = cv2.filter2D(inputimg, -1, mask, borderType=cv2.BORDER_REPLICATE)

    return out

def fourrier_transform(img):
    # Fourier
    #f = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    f = np.fft.fft2(img)
    # Fourier Shift - Put center zero.
    fs = np.fft.fftshift(f)

    return fs

def magnitude(fourier_shift):
    #return np.log(1 + cv2.magnitude(fourier_shift[:, :, 0],fourier_shift[:, :, 1]))
    return np.abs(fourier_shift)   
    #return 20*np.log(np.abs(fourier_shift))
    #return 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))

def phase(fourier):
    return np.unwrap(np.angle(fourier))

def inverse_fourier_transform(fourier_shift, percentage = 100):
    f_ishift = np.fft.ifftshift(fourier_shift)
    img_back = np.fft.ifft2(f_ishift)
    return magnitude(img_back)

