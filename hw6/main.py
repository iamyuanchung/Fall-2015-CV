import sys

import numpy as np
import cv2


def main():
    img = cv2.imread('lena.bmp', 0)
    # img is now a 512 x 512 numpy.ndarray

    # binarize the image first ...
    img_bin = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img[i][j] >= 128:
                img_bin[i][j] = 1

    # downsample lena image from 512x512 to 64x64 by
    # using 8x8 blocks as a unit and take the
    # topmost-left pixel as the downsampled data ...
    img_down = np.zeros((64, 64), np.int)
    for i in xrange(img_down.shape[0]):
        for j in xrange(img_down.shape[1]):
            img_down[i][j] = img_bin[8 * i][8 * j]

    # define the operation function for
    # Yokoi Connectivity Number ...
    def h(b, c, d, e):
        if b == c and (d != b or e != b):
            return 'q'
        if b == c and (d == b and e == b):
            return 'r'
        return 's'

    # compute and output Yokoi Connectivity Number ...
    for i in xrange(img_down.shape[0]):
        for j in xrange(img_down.shape[1]):
            if img_down[i][j] > 0:  # foreground pixel
                if i == 0:
                    if j == 0:
                    # top-left
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, img_down[i][j], img_down[i][j + 1]
                        x8, x4, x5 = 0, img_down[i + 1][j], img_down[i + 1][j + 1]
                    elif j == img_down.shape[1] - 1:
                    # top-right
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img_down[i][j - 1], img_down[i][j], 0
                        x8, x4, x5 = img_down[i + 1][j - 1], img_down[i + 1][j], 0
                    else:
                    # top-row
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img_down[i][j - 1], img_down[i][j], img_down[i][j + 1]
                        x8, x4, x5 = img_down[i + 1][j - 1], img_down[i + 1][j], img_down[i + 1][j + 1]
                elif i == img_down.shape[0] - 1:
                    if j == 0:
                    # bottom-left
                        x7, x2, x6 = 0, img_down[i - 1][j], img_down[i - 1][j + 1]
                        x3, x0, x1 = 0, img_down[i][j], img_down[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                    elif j == img_down.shape[1] - 1:
                    # bottom-right
                        x7, x2, x6 = img_down[i - 1][j - 1], img_down[i - 1][j], 0
                        x3, x0, x1 = img_down[i][j - 1], img_down[i][j], 0
                        x8, x4, x5 = 0, 0, 0
                    else:
                    # bottom-row
                        x7, x2, x6 = img_down[i - 1][j - 1], img_down[i - 1][j], img_down[i - 1][j + 1]
                        x3, x0, x1 = img_down[i][j - 1], img_down[i][j], img_down[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                else:
                    if j == 0:
                        x7, x2, x6 = 0, img_down[i - 1][j], img_down[i - 1][j + 1]
                        x3, x0, x1 = 0, img_down[i][j], img_down[i][j + 1]
                        x8, x4, x5 = 0, img_down[i + 1][j], img_down[i + 1][j + 1]
                    elif j == img_down.shape[1] - 1:
                        x7, x2, x6 = img_down[i - 1][j - 1], img_down[i - 1][j], 0
                        x3, x0, x1 = img_down[i][j - 1], img_down[i][j], 0
                        x8, x4, x5 = img_down[i + 1][j - 1], img_down[i + 1][j], 0
                    else:
                        x7, x2, x6 = img_down[i - 1][j - 1], img_down[i - 1][j], img_down[i - 1][j + 1]
                        x3, x0, x1 = img_down[i][j - 1], img_down[i][j], img_down[i][j + 1]
                        x8, x4, x5 = img_down[i + 1][j - 1], img_down[i + 1][j], img_down[i + 1][j + 1]

                a1 = h(x0, x1, x6, x2)
                a2 = h(x0, x2, x7, x3)
                a3 = h(x0, x3, x8, x4)
                a4 = h(x0, x4, x5, x1)

                if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
                    ans = 5
                else:
                    ans = 0
                    for a_i in [a1, a2, a3, a4]:
                        if a_i == 'q':
                            ans += 1

                print '%d' % ans,

            else:
                print ' ',

            if j == img_down.shape[1] - 1:
                print ''


if __name__ == '__main__':
    main()
