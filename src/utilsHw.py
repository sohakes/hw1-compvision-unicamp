import cv2
import numpy as np
import math
import time

DEBUG = True    

def merge_color(p_b,p_g,p_r, n_access):
    access_b = p_b.access(n_access)
    access_r = p_r.access(n_access)
    access_g = p_g.access(n_access)
    return cv2.merge((access_b, access_g, access_r))   

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

def create_circular_mask(width, height, centerx, centery, radius):

    y,x = np.ogrid[-centerx:width-centerx, -centery:height-centery]
    mask = x*x + y*y <= radius*radius

    array = np.zeros((width, height))
    array[mask] = 255

    return array.astype('uint8')

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
    for i in range(pad, inputimg.shape[0] + pad):
        for j in range(pad, inputimg.shape[1] + pad):
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
    print("shapefs", fs.shape)

    return fs

def magnitude(fourier_shift):
    #return np.log(1 + cv2.magnitude(fourier_shift[:, :, 0],fourier_shift[:, :, 1]))
    return np.abs(fourier_shift)
    #return 20*np.log(np.abs(fourier_shift))
    #return (cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))

def phase(fourier):
    #return (cv2.phase(fourier[:,:,0],fourier[:,:,1]))
    return (np.angle(fourier))

def remove_freq(fourier, radius):
    fourier = fourier.copy()
    mask = create_circular_mask(fourier.shape[0], fourier.shape[1], fourier.shape[0]/2, fourier.shape[1]/2, radius).astype(float) / 255
    fourier = fourier * mask

    return fourier

def remove_phase(fourier, percup, percdown):
    fourier = fourier.copy()
    phases = np.angle(fourier)
    filtered_valup = np.min(phases[np.nonzero(phases)])
    filtered_valdown = np.max(phases)
    if (percup != -1):
        filtered_valup = np.percentile(phases, percup)
    if (percdown != -1):
        filtered_valdown = np.percentile(phases, percdown)   
        
    fourier[phases > filtered_valup] = 0             
    fourier[phases < filtered_valdown] = 0

    return fourier

def remove_magnitude(fourier, percup, percdown):
    fourier = fourier.copy()
    fourier_abs = np.abs(fourier)
    filtered_valup = np.min(fourier_abs[np.nonzero(fourier_abs)])
    filtered_valdown = np.max(fourier_abs)
    if (percup != -1):
        filtered_valup = np.percentile(fourier_abs, percup)
    if (percdown != -1):
        filtered_valdown = np.percentile(fourier_abs, percdown)  
    fourier[fourier_abs > filtered_valup] = 0
    fourier[fourier_abs < filtered_valdown] = 0
    return fourier

def inverse_fourier_transform(fourier_shift, percentage_phase_up = 100.0, percentage_magnitude_up = 100.0,
     percentage_phase_down = 0.0, percentage_magnitude_down = 0.0):
    """
        This function works like this: if you don't change the percentages, it will simply recover the original image.
        If you change the up percentages, it will zero everyone bigger than that. 
        If you change the down percentages, it will zero everyone smaller.
        If you change anything to -1, it will get the min or max pixel (depending if it's up or down)
    """
    



    fourier_shift = fourier_shift.copy()

    fourier_shift = remove_phase(fourier_shift,percentage_phase_up, percentage_phase_down)
    fourier_shift = remove_magnitude(fourier_shift,percentage_magnitude_up, percentage_magnitude_down)

    f_ishift = np.fft.ifftshift(fourier_shift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    return img_back.astype('uint8')


def frequency_blend(img1, img2, mask):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    mask = (mask.astype(float) / 255).astype(float)
    img1mask = img1 * mask
    img2mask = img2 * (1 - mask)

    f1l = fourrier_transform(img1mask)
    f2l = f1l
    f1x = f1l
    f1x = remove_phase(f1x, 80, 0)
    f1x = remove_magnitude(f1x, 100, 0)
    debug('test', inverse_fourier_transform(f1x).astype('uint8'))
    f1x = f1l
    f1x = remove_phase(f1x, 100, 20)
    f1x = remove_magnitude(f1x, 100, 0)
    debug('test', inverse_fourier_transform(f1x).astype('uint8'))
    f1x = f1l
    f1x = remove_phase(f1x, 100, 0)
    f1x = remove_magnitude(f1x, 99.95, 0)
    debug('test', inverse_fourier_transform(f1x).astype('uint8'))
    f1x = f1l
    f1x = remove_phase(f1x, 100, 0)
    f1x = remove_magnitude(f1x, 100, 70.0)
    debug('test', inverse_fourier_transform(f1x).astype('uint8'))

    f1l = fourrier_transform(img2mask)
    f1x = f1l
    f1x = remove_phase(f1x, 80, 0)
    f1x = remove_magnitude(f1x, 100, 0)
    debug('test', inverse_fourier_transform(f1x).astype('uint8'))
    f1x = f1l
    f1x = remove_phase(f1x, 100, 20)
    f1x = remove_magnitude(f1x, 100, 0)
    debug('test', inverse_fourier_transform(f1x).astype('uint8'))
    f1x = f1l
    f1x = remove_phase(f1x, 100, 0)
    f1x = remove_magnitude(f1x, 99.95, 0)
    f1x2 = remove_magnitude(f1l, 100, 0.05)
    debug('test', inverse_fourier_transform(f1x + f1x2).astype('uint8'))
    f1x = f1l
    f1x = remove_phase(f1x, 100, 0)
    f1x = remove_magnitude(f1x, 100, 70.0)
    debug('test', inverse_fourier_transform(f1x).astype('uint8'))
    debug('test', inverse_fourier_transform(remove_freq(f1l, 10) - f1l).astype('uint8'))
    debug('test', inverse_fourier_transform(f1l - remove_freq(f1l, 10)).astype('uint8'))
    debug('test', inverse_fourier_transform(remove_freq(f1l, 10) + remove_freq(f2l, 10)).astype('uint8'))
    debug('test', inverse_fourier_transform(remove_freq(f1l, 50) + remove_freq(f2l, 50)).astype('uint8'))
    debug('test', inverse_fourier_transform(remove_freq(f1l, 100) + remove_freq(f2l, 100)).astype('uint8'))
    debug('test', inverse_fourier_transform(remove_freq(f1l, 200) + remove_freq(f2l, 200)).astype('uint8'))
    debug('test', inverse_fourier_transform(remove_freq(f1x, 50)).astype('uint8'))
    debug('test', inverse_fourier_transform(remove_freq(f1x, 100)).astype('uint8'))
    debug('test', inverse_fourier_transform(remove_freq(f1x, 200)).astype('uint8'))

    f1 = fourrier_transform(img1mask)
    f1 = remove_phase(f1, 100.0, 0.0)
    f1 = remove_magnitude(f1, 99.95, 0.0)
    f2 = fourrier_transform(img2mask)
    f2 = remove_phase(f2, 100.0, 0.0)
    f2 = remove_magnitude(f2, 100.0, 0.05)

    f3 = f1 + f2
    return inverse_fourier_transform(f3)