import cv2
import numpy as np
import math
import time

################  HW1  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################

DEBUG = True

def debug(name,img):
    if DEBUG == False:
        return

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#got it from https://stackoverflow.com/questions/5478351/python-time-measure-function
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap

#debug('noisy_imgg',noisy_imgg)
#cv2.imwrite('output/p0-5-b-0.png', noisy_imgb)

#!!!
#IMPORTANTE: ACHO QUE PRECISA FUNCIONAR COM IMAGEM COLORIDA!!!
#!!!
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

img = cv2.imread('input/p1-1-0.png', cv2.IMREAD_GRAYSCALE)
conv = convolution(img, np.array([[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]]))
debug('conv',conv)
cv2.imwrite('output/p1-2-1-0.png', conv)
convocv = convolution_opencv(img, np.array([[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]]))
debug('convocv',convocv)
cv2.imwrite('output/p1-2-1-1.png', convocv)

def create_gaussian_mask(size, sigma):
    assert size % 2 == 1, "mask should have odd size"
    def pixel_val(x, y):
        #return np.exp(-(X.^2 + Y.^2) / (2*sigma*sigma));
        return (1.0/(2 * math.pi * sigma ** 2)) * math.e**(-(x**2 + y**2)/(2*sigma**2))

    halfsize = math.floor(size / 2)

    mask = np.array([[pixel_val(i, j) for i in range(-halfsize, halfsize + 1)] for j in range(-halfsize, halfsize + 1)])
    msum = np.sum(mask)

    return mask / msum

gaussian_mask = create_gaussian_mask(5, 3)
gaussian_img = convolution(img, gaussian_mask)
debug('gaussian_img',gaussian_img)
cv2.imwrite('output/p1-2-2-0.png', gaussian_img)

class Pyramid:
    def __up(self, img): #downsample
        img = convolution(img, self.__gaussian_mask)
        return img[::2, ::2]

        #private
    def __down(self, img): #upsample
        #first x direction
        upimg = img.astype(int)
        #double the last column
        upimg = np.insert(upimg, img.shape[1] - 1, upimg[:, -1], axis=1)
        for i in range(img.shape[1], 0, -1):
            result_column = (upimg[:, i] + upimg[:, i - 1]) / 2
            upimg = np.insert(upimg, i, result_column, axis=1)

        #double the last row
        upimg = np.insert(upimg, img.shape[0] - 1, upimg[-1, :], axis=0)
        for i in range(img.shape[0], 0, -1):
            result_row = (upimg[i, :] + upimg[i - 1, :]) / 2
            upimg = np.insert(upimg, i, result_row, axis=0)
        print(upimg)
        return upimg.astype(np.uint8)

    #public
    def access(self, level):
        return self._img_arr[level]

    def __init__(self, image, levels):
        self.__gaussian_mask = create_gaussian_mask(5, 3)
        self.image = image
        self.levels = levels
        self.__img_arr = [image]
        debug("down", self.__down(image))
        for i in range(levels - 1):
          self.__img_arr.append(self.__up(self.__img_arr[i]))
          debug("up", self.__img_arr[-1])





#downsampled_img = downsample(gaussian_img)
#debug('gaussian_img',downsampled_img)
#cv2.imwrite('output/p1-2-2-1.png', downsampled_img)

pyramid = Pyramid(img, 3)
