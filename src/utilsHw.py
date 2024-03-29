import cv2
import numpy as np
import math
import time

DEBUG = False
NUMBER_FILE = -1

# define number file for convolution
def numFile():
    global NUMBER_FILE
    NUMBER_FILE = NUMBER_FILE + 1
    return NUMBER_FILE 

def defineMask(r,g,b, mask_type, file_name):
    if (mask_type == 'circle'):
        mask_r = create_circular_mask(r.shape[0], r.shape[1], 220, 220, 200)
        mask_g = create_circular_mask(g.shape[0], g.shape[1], 220, 220, 200)
        mask_b = create_circular_mask(b.shape[0], b.shape[1], 220, 220, 200)
        return r,g,b
    else:
        mask = cv2.imread('input/'+ str(file_name))
        bm,gm,rm = cv2.split(mask)
        rm = np.pad(rm, ((0, 1), (0, 1)), 'edge')
        gm = np.pad(gm, ((0, 1), (0, 1)), 'edge')
        bm = np.pad(bm, ((0, 1), (0, 1)), 'edge')    
        return rm,gm,bm    

def merge_color(p_b,p_g,p_r, n_access):
    access_b = p_b.access(n_access)
    access_r = p_r.access(n_access)
    access_g = p_g.access(n_access)
    return cv2.merge((access_b, access_g, access_r))   

def debug_print(val):
   if DEBUG == False:
        return
   print(val)

def debug(name,img):
    if DEBUG == False:
        return

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_gaussian_mask_fourrier(size, sigma, imgsize):
    assert size % 2 == 1, "mask should have odd size"
    def pixel_val(x, y):
        return (1.0/(2 * math.pi * sigma ** 2)) * math.e**(-(x**2 + y**2)/(2*sigma**2))
    
    halfsize = math.floor(size / 2)

    mask = np.array([[pixel_val(i, j) for i in range(-halfsize, halfsize + 1)] for j in range(-halfsize, halfsize + 1)]).astype(float)
    msum = float(np.sum(mask))
    mask = mask/msum

    nmask = np.zeros((imgsize, imgsize)).astype(float)
    middle = int(imgsize/2)
    middlemask = int(size/2)
    nmask[middle-middlemask:middle+middlemask+1, middle-middlemask:middle+middlemask+1] = mask

    return nmask


def create_gaussian_mask(size, sigma):
    assert size % 2 == 1, "mask should have odd size"
    def pixel_val(x, y):
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
        debug_print(str('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0)))
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
    f = np.fft.fft2(img)
    # Fourier Shift - Put center zero.
    fs = np.fft.fftshift(f)
    debug_print(str("shapefs " + str(fs.shape)))

    return fs

def magnitude(fourier_shift):
    return np.abs(fourier_shift)

def phase(fourier):
    return (np.angle(fourier))

def remove_freq(fourier, radius, inv=False):
    fourier = fourier.copy()
    mask = create_circular_mask(fourier.shape[0], fourier.shape[1], fourier.shape[0]/2, fourier.shape[1]/2, radius).astype(float) / 255
    if inv:
        mask = 1 - mask
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
        percdown = 100 - percdown
        filtered_valdown = np.percentile(phases, percdown)   
        
    def apply_filter(vphase, v):
        if vphase > filtered_valup or vphase < filtered_valdown:
            return np.abs(v) * np.exp(1j*0)
        return v
    vecapply = np.vectorize(apply_filter)
    fourier = vecapply(phases, fourier)

    return fourier

def remove_magnitude(fourier, percup, percdown):
    fourier = fourier.copy()
    fourier_abs = np.abs(fourier)
    filtered_valup = np.min(fourier_abs[np.nonzero(fourier_abs)])
    filtered_valdown = np.max(fourier_abs)
    if (percup != -1):
        filtered_valup = np.percentile(fourier_abs, percup)
    if (percdown != -1):
        percdown = 100 - percdown
        filtered_valdown = np.percentile(fourier_abs, percdown)  

    def apply_filter(vabs, v):
        if vabs > filtered_valup or vabs < filtered_valdown:
            return 0 * np.exp(1j*np.angle(v))
            #return np.complex(np.angle(v))
        return v
    vecapply = np.vectorize(apply_filter)
    fourier = vecapply(fourier_abs, fourier)
    return fourier

def inverse_fourier_transform(fourier_shift, percentage_phase_up = 100.0, percentage_magnitude_up = 100.0,
     percentage_phase_down = 100.0, percentage_magnitude_down = 100.0):
    """
        This function works like this: if you don't change the percentages, it will simply recover the original image.
        If you change the up percentages, it will zero everyone bigger than that. 
        If you change the down percentages, it will zero everyone smaller, but starting from the biggest ones (so 75 would remove the 25 lower).
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
    assert img1.shape == img2.shape == mask.shape
    img1mask = img1 * mask
    img2mask = img2 * (1 - mask)
    size = img1.shape[0]

    f1o = fourrier_transform(img1mask)
    f2o = fourrier_transform(img2mask)
    f1, f2 = f1o, f2o

    #simple sum
    f3 = f1 + f2
    yield inverse_fourier_transform(f3)


    #filter
    f1, f2 = f1o, f2o
    f3 = f1 + f2
    yield inverse_fourier_transform(remove_freq(f3, 15))

    #gaussian
    f1, f2 = f1o, f2o
    g_mask = fourrier_transform(create_gaussian_mask_fourrier(31, 6, size))
    f1abs = np.abs(f1) * np.abs(g_mask)
    f1 = f1abs * np.exp(1j*np.angle(f1))
    f2abs = np.abs(f2) * np.abs(g_mask)
    f2 = f1abs * np.exp(1j*np.angle(f2))
    f3 = f1 + f2
    yield inverse_fourier_transform(f3)

    #filter
    f1, f2 = f1o, f2o
    f3 = remove_magnitude(f1, 100, 1) + remove_magnitude(f2, 100, 1)
    yield inverse_fourier_transform(f3)
