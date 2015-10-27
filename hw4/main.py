import sys

import numpy as np
import cv2


def dilation(img_bin, kernel):
    img_dil = np.zeros(img_bin.shape, np.int)
    for i in xrange(img_bin.shape[0]):
        for j in xrange(img_bin.shape[1]):
            if img_bin[i][j] > 0:
                for element in kernel:
                    p, q = element
                    if (i + p) >= 0 and (i + p) <= (img_bin.shape[0] - 1) and   \
                       (j + q) >= 0 and (j + q) <= (img_bin.shape[1] - 1):
                        img_dil[i + p][j + q] = 255
    return img_dil

def erosion(img_bin, kernel):
    img_ero = np.zeros(img_bin.shape, np.int)
    for i in xrange(img_bin.shape[0]):
        for j in xrange(img_bin.shape[1]):
            if img_bin[i][j] > 0:
                exist = True
                for element in kernel:
                    p, q = element
                    if (i + p) < 0 or (i + p) > (img_bin.shape[0] - 1) or   \
                       (j + q) < 0 or (j + q) > (img_bin.shape[1] - 1) or   \
                       img_bin[i + p][j + q] == 0:
                        exist = False
                        break
                if exist:
                    img_ero[i][j] = 255
    return img_ero

def closing(img_bin, kernel):
    return erosion(dilation(img_bin, kernel), kernel)

def opening(img_bin, kernel):
    return dilation(erosion(img_bin, kernel), kernel)

def hit_and_miss(img_bin, J_kernel, K_kernel):
    # img_comp is the complement of img_bin
    img_comp = -img_bin + 255
    return (erosion(img_bin, J_kernel) + erosion(img_comp, K_kernel)) / 2

def main():
    img = cv2.imread('lena.bmp', 0)
    # img is now a 512 x 512 numpy.ndarray

    # binarize the image first ...
    img_bin = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img[i][j] >= 128:
                img_bin[i][j] = 255
    # output the binarized image ...
    cv2.imwrite('lena.bin.bmp', img_bin)

    # kernel is a 3-5-5-5-3 octagon, where
    # the orgin is at the center
    kernel = [[-2, -1], [-2, 0], [-2, 1],
              [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
              [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
              [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
              [2, -1], [2, 0], [2, 1]]

    # perform binary morphological dilation
    print 'performing binary morphological dilation ...\n'
    img_dil = dilation(img_bin, kernel)
    cv2.imwrite('lena.bin.dil.bmp', img_dil)

    # perform binary morphological erosion
    print 'performing binary morphological erosion ...\n'
    img_ero = erosion(img_bin, kernel)
    cv2.imwrite('lena.bin.ero.bmp', img_ero)

    # perform binary morphological closing
    print 'performing binary morphological closing ...\n'
    img_close = closing(img_bin, kernel)
    cv2.imwrite('lena.bin.close.bmp', img_close)

    # perform binary morphological opening
    print 'performing binary morphological opening ...\n'
    img_open = opening(img_bin, kernel)
    cv2.imwrite('lena.bin.open.bmp', img_open)

    # kernels for hit-and-miss
    J_kernel = [[0, -1], [0, 0], [1, 0]]
    K_kernel = [[-1, 0], [-1, 1], [0, 1]]

    # perform hit-and-miss transformion
    print 'performing hit-and-miss transformation ...\n'
    img_ham = hit_and_miss(img_bin, J_kernel, K_kernel)
    cv2.imwrite('lena.bin.ham.bmp', img_ham)

    print 'All tasks are done.'


if __name__ == '__main__':
    main()
