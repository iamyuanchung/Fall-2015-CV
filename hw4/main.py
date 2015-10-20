import sys

import numpy as np
import cv2


def main():
    img = cv2.imread('lena.bmp', 0) # img is a 512 x 512 numpy.ndarray

    # binarize the image first ...
    img_bin = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            # if img[i][j] >= 128
            if img[i][j] > 128:
                img_bin[i][j] = 255

    cv2.imwrite('lena.bin.bmp', img_bin)

    # perform binary dilation by structuring element
    img_dil = np.zeros(img.shape, np.int)

    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img_bin[i][j] > 0:
                img_dil[i][j] = 255
                if i == 0:
                    if j == 0:
                        # top-left
                        img_dil[i][j + 1] = 255
                        img_dil[i + 1][j] = 255
                    elif j == img.shape[1] - 1:
                        # top-right
                        img_dil[i][j - 1] = 255
                        img_dil[i + 1][j] = 255
                    else:
                        img_dil[i][j - 1] = 255
                        img_dil[i + 1][j] = 255
                        img_dil[i][j + 1] = 255
                elif i == img.shape[0] - 1:
                    if j == 0:
                        # bottom-left
                        img_dil[i - 1][j] = 255
                        img_dil[i][j + 1] = 255
                    elif j == img.shape[1] - 1:
                        # bottom-right
                        img_dil[i][j - 1] = 255
                        img_dil[i - 1][j] = 255
                    else:
                        img_dil[i][j - 1] = 255
                        img_dil[i - 1][j] = 255
                        img_dil[i][j + 1] = 255
                else:
                    if j == 0:
                        img_dil[i - 1][j] = 255
                        img_dil[i][j + 1] = 255
                        img_dil[i + 1][j] = 255
                    elif j == img.shape[1] - 1:
                        img_dil[i - 1][j] = 255
                        img_dil[i][j - 1] = 255
                        img_dil[i + 1][j] = 255
                    else:
                        img_dil[i - 1][j] = 255
                        img_dil[i][j + 1] = 255
                        img_dil[i + 1][j] = 255
                        img_dil[i][j - 1] = 255
    
    # output lena.bin.dil ...
    cv2.imwrite('lena.bin.dil.bmp', img_dil)


if __name__ == '__main__':
    main()
