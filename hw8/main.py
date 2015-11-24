import copy

import numpy as np
import cv2


def main():
    img = cv2.imread('lena.bmp', 0)
    # img is now a 512 x 512 numpy.ndarray

    # Generate and output an image with
    # additive white Gaussian noise
    # with amplitude = 10
    img_gauss_10 = img + 10 * np.random.normal(0, 1, img.shape)
    cv2.imwrite('lena.gaussian.10.bmp', img_gauss_10)

    # Generate and output an image with
    # additive white Gaussian noise
    # with amplitude = 30
    img_gauss_30 = img + 30 * np.random.normal(0, 1, img.shape)
    cv2.imwrite('lena.gaussian.30.bmp', img_gauss_30)

    # Generate and output an image with
    # salt-and-pepper noise with
    # threshold = 0.05
    prob_map = np.random.uniform(0, 1, img.shape)
    img_sp_5 = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if prob_map[i][j] < 0.05:
                img_sp_5[i][j] = 0
            elif prob_map[i][j] > 1 - 0.05:
                img_sp_5[i][j] = 255
            else:
                img_sp_5[i][j] = img[i][j]
    cv2.imwrite('lena.sp.05.bmp', img_sp_5)

    # Generate and output an image with
    # salt-and-pepper noise with
    # threshold = 0.1
    prob_map = np.random.uniform(0, 1, img.shape)
    img_sp_10 = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if prob_map[i][j] < 0.1:
                img_sp_10[i][j] = 0
            elif prob_map[i][j] > 1 - 0.1:
                img_sp_10[i][j] = 255
            else:
                img_sp_10[i][j] = img[i][j]
    cv2.imwrite('lena.sp.10.bmp', img_sp_10)

    


if __name__ == '__main__':
    main()
