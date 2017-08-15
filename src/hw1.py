import cv2
import numpy as np
import math

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

#debug('noisy_imgg',noisy_imgg)
#cv2.imwrite('output/p0-5-b-0.png', noisy_imgb)

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

img = cv2.imread('input/p1-1-0.png', cv2.IMREAD_GRAYSCALE)
conv = convolution(img, np.array([[0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1]]))
debug('conv',conv)
cv2.imwrite('output/p1-1-2-0.png', conv)
