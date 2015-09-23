import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    # usage: python ./main.py
    # load lena.bmp into a 2D numpy.array
    img = cv2.imread('lena.bmp', 0)

    # TODO: binarize the image (threshold at 128)
    img_bin = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img[i][j] > 128:
                img_bin[i][j] = 255
    cv2.imwrite('binarize.bmp', img_bin)

    # TODO: generate histogram
    # flow: calculate distribution
    #       -> write to dist.csv file
    #       -> plot with excel
    # the resulting image is hist.png
    dist = np.zeros(256, np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            dist[img[i][j]] += 1
    np.savetxt("dist.csv", dist, delimiter=",")

    # TODO: connected components (regions with + at centroid, bounding box)



if __name__ == '__main__':
    main()
