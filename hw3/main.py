import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    # usage: python ./main.py [image_file]
    # image_file is the specified source image,
    # and is lena.bmp by default.
    assert len(sys.argv) <= 2

    if len(sys.argv) == 2:
        image_file = sys.argv[1]
    else:
        image_file = 'lena.bmp'

    img = cv2.imread(image_file, 0)

    # divide image pixels by 3 and output the image
    img /= 3
    cv2.imwrite('lena_divided_3.bmp', img)

    # compute the distribution for histogram, i.e. r_{k}
    dist = np.zeros(256, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            dist[img[i][j]] += 1

    # output the histogram of original image
    # TODO: need to specify the trick here in report ...
    fig = plt.figure()
    plt.bar(np.arange(257), np.append(dist, np.array([1])), color='r')
    plt.savefig('histogram-before-equalized.png')

    # compute s_{k}
    dist_accumulator = np.zeros(256, np.int)
    dist_accumulator[0] = dist[0]
    for i in xrange(1, 256):
        dist_accumulator[i] = dist_accumulator[i - 1] + dist[i]

    dist_accumulator = dist_accumulator * 255. / np.sum(dist)

    img_he = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            img_he[i][j] = dist_accumulator[img[i][j]]

    dist_he = np.zeros(256, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            dist_he[img_he[i][j]] += 1

    # output the histogram of equalized image
    fig = plt.figure()
    plt.bar(np.arange(256), dist_he, color='r')
    plt.savefig('histogram-after-equalized.png')

    # output the resulting image
    cv2.imwrite('lena_equalized.bmp', img_he)


if __name__ == '__main__':
    main()
