import copy

import numpy as np
import cv2

from utils import mark_interior_border,     \
                  mark_pair_relationship,   \
                  compute_yokoi_number


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
    # cv2.imwrite('lena.bin.bmp', img_bin)

    # downsample lena image from 512x512 to 64x64 by
    # using 8x8 blocks as a unit and take the
    # topmost-left pixel as the downsampled data ...
    img_down = np.zeros((64, 64), np.int)
    for i in xrange(img_down.shape[0]):
        for j in xrange(img_down.shape[1]):
            img_down[i][j] = img_bin[8 * i][8 * j]

    # output the downsampled binarized image ...
    # cv2.imwrite('lena.bin.down.bmp', img_down)

    img_thin = img_down
    # Thinning operator ...
    while True:
        img_thin_old = copy.deepcopy(img_thin)

        # Step1 - mark the interior/border pixels
        # input: original symbolic image
        # output: interior/border-marked image
        img_ib = mark_interior_border(img_thin)

        # Step 2 - pair relationship operator
        # input: interior/border-marked image
        # output: pair-marked image
        img_pair = mark_pair_relationship(img_ib)

        # Step 3 - check and delete the deletable pixels
        # input: original symbolic image and
        #        pair-marked image
        # output: thinned image
        yokoi_map = compute_yokoi_number(img_thin)
        delete_map = (yokoi_map == 1) * 1
        for i in xrange(img_pair.shape[0]):
            for j in xrange(img_pair.shape[1]):
                if delete_map[i][j] == 1 and    \
                   img_pair[i][j] == 1: # 'p'
                    img_thin[i][j] = 0

        # use thinned output image as the next original
        # symbolic image and repeat the abovementioned
        # 3 steps until the last output stops changing
        if np.sum(img_thin == img_thin_old) == img_thin.shape[0] * img_thin.shape[1]:
            break

    cv2.imwrite('lena.thinned.bmp', img_thin)


if __name__ == '__main__':
    main()
