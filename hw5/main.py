import sys

import numpy as np
import cv2


def dilation(img, kernel):
    img_dil = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img[i][j] > 0:
                max_value = 0
                for element in kernel:
                    p, q = element
                    if (i + p) >= 0 and (i + p) <= (img.shape[0] - 1) and   \
                       (j + q) >= 0 and (j + q) <= (img.shape[1] - 1):
                        if img[i + p][j + q] > max_value:
                            max_value = img[i + p][j + q]
                for element in kernel:
                    p, q = element
                    if (i + p) >= 0 and (i + p) <= (img.shape[0] - 1) and   \
                       (j + q) >= 0 and (j + q) <= (img.shape[1] - 1):
                        img_dil[i + p][j + q] = max_value
    return img_dil

def erosion(img, kernel):
    img_ero = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img[i][j] > 0:
                exist = True
                min_value = np.inf
                for element in kernel:
                    p, q = element
                    if (i + p) >= 0 and (i + p) <= (img.shape[0] - 1) and   \
                       (j + q) >= 0 and (j + q) <= (img.shape[1] - 1):
                        if img[i + p][j + q] == 0:
                            exist = False
                            break
                        if img[i + p][j + q] < min_value:
                            min_value = img[i + p][j + q]
                exist = True
                for element in kernel:
                    p, q = element
                    if (i + p) >= 0 and (i + p) <= (img.shape[0] - 1) and   \
                       (j + q) >= 0 and (j + q) <= (img.shape[1] - 1):
                        if img[i + p][j + q] == 0:
                            exist = False
                            break
                    if (i + p) >= 0 and (i + p) <= (img.shape[0] - 1) and   \
                       (j + q) >= 0 and (j + q) <= (img.shape[1] - 1) and   \
                       exist:
                        img_ero[i + p][j + q] = min_value
    return img_ero

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def main():
    img = cv2.imread('lena.bmp', 0)
    # img is now a 512 x 512 numpy.ndarray

    # kernel is a 3-5-5-5-3 octagon, where
    # the orgin is at the center
    kernel = [
        [-2, -1], [-2, 0], [-2, 1],
        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
        [2, -1], [2, 0], [2, 1]
    ]

    # perform grayscale morphological dilation
    print 'performing grayscale morphological dilation ...\n'
    img_dil = dilation(img, kernel)
    cv2.imwrite('lena.gray.dil.bmp', img_dil)

    # perform grayscale morphological erosion
    print 'performing grayscale morphological erosion ...\n'
    img_ero = erosion(img, kernel)
    cv2.imwrite('lena.gray.ero.bmp', img_ero)

    # perform grayscale morphological closing
    print 'performing grayscale morphological closing ...\n'
    img_close = closing(img, kernel)
    cv2.imwrite('lena.gray.close.bmp', img_close)

    # perform grayscale morphological opening
    print 'performing grayscale morphological opening ...\n'
    img_open = opening(img, kernel)
    cv2.imwrite('lena.gray.open.bmp', img_open)

    print 'All tasks are done.'


if __name__ == '__main__':
    main()
