import copy

import numpy as np
import cv2

from utils import opening, closing


def generate_Gaussian_noise(img, mu, sigma, amplitude):
    # The function generates Gaussian noise with the specified
    # mu, sigma, and amplitude on the given image.
    return img + amplitude * np.random.normal(mu, sigma, img.shape)

def generate_salt_and_pepper_noise(img, low, high, threshold):
    # The function generates salt-and-pepper noise with the
    # specified range for uniform sampling and threshold on
    # the given image.
    prob_map = np.random.uniform(low, high, img.shape)
    img_sp = copy.deepcopy(img)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if prob_map[i][j] < threshold:
                img_sp[i][j] = 0
            elif prob_map[i][j] > 1 - threshold:
                img_sp[i][j] = 255
    return img_sp

def box_filter(img, filter_size):
    # The function runs box filter with the specified
    # filter_size on the given image.
    img_fil = np.zeros(
        shape=(img.shape[0] - filter_size, img.shape[1] - filter_size),
        dtype=np.int
    )
    for i in xrange(img_fil.shape[0]):
        for j in xrange(img_fil.shape[1]):
            img_fil[i][j] = np.mean(img[i: i + filter_size, j: j + filter_size])
    return img_fil

def median_filter(img, filter_size):
    # The function median filter with the specified
    # filter_size on the given image.
    img_fil = np.zeros(
        shape=(img.shape[0] - filter_size, img.shape[1] - filter_size),
        dtype=np.int
    )
    for i in xrange(img_fil.shape[0]):
        for j in xrange(img_fil.shape[1]):
            img_fil[i][j] = np.median(img[i: i + filter_size, j: j + filter_size])
    return img_fil

def main():
    img = cv2.imread('lena.bmp', 0)
    # img is now a 512 x 512 numpy.ndarray

    # Generate and output an image with
    # additive white Gaussian noise
    # with amplitude = 10
    img_gauss_10 = generate_Gaussian_noise(img, 0, 1, 10)
    cv2.imwrite('lena.gaussian.10.bmp', img_gauss_10)

    # Generate and output an image with
    # additive white Gaussian noise
    # with amplitude = 30
    img_gauss_30 = generate_Gaussian_noise(img, 0, 1, 30)
    cv2.imwrite('lena.gaussian.30.bmp', img_gauss_30)

    # Generate and output an image with
    # salt-and-pepper noise with
    # threshold = 0.05
    img_sp_05 = generate_salt_and_pepper_noise(img, 0, 1, 0.05)
    cv2.imwrite('lena.sp.05.bmp', img_sp_05)

    # Generate and output an image with
    # salt-and-pepper noise with
    # threshold = 0.1
    img_sp_10 = generate_salt_and_pepper_noise(img, 0, 1, 0.1)
    cv2.imwrite('lena.sp.10.bmp', img_sp_10)

    # Run 3x3 box filter on the image
    # with white Gaussian noise with
    # amplitude = 10
    img_gauss_10_box_3 = box_filter(img_gauss_10, 3)
    cv2.imwrite('lena.gaussian.10.box.3x3.bmp', img_gauss_10_box_3)

    # Run 3x3 box filter on the image
    # with white Gaussian noise with
    # amplitude = 30
    img_gauss_30_box_3 = box_filter(img_gauss_30, 3)
    cv2.imwrite('lena.gaussian.30.box.3x3.bmp', img_gauss_30_box_3)

    # Run 3x3 box filter on the image
    # with salt-and-pepper noise with
    # threshold = 0.05
    img_sp_05_box_3 = box_filter(img_sp_05, 3)
    cv2.imwrite('lena.sp.05.box.3x3.bmp', img_sp_05_box_3)

    # Run 3x3 box filter on the image
    # with salt-and-pepper noise with
    # threshold = 0.1
    img_sp_10_box_3 = box_filter(img_sp_10, 3)
    cv2.imwrite('lena.sp.10.box.3x3.bmp', img_sp_10_box_3)

    # Run 5x5 box filter on the image
    # with white Gaussian noise with
    # amplitude = 10
    img_gauss_10_box_5 = box_filter(img_gauss_10, 5)
    cv2.imwrite('lena.gaussian.10.box.5x5.bmp', img_gauss_10_box_5)

    # Run 5x5 box filter on the image
    # with white Gaussian noise with
    # amplitude = 30
    img_gauss_30_box_5 = box_filter(img_gauss_30, 5)
    cv2.imwrite('lena.gaussian.30.box.5x5.bmp', img_gauss_30_box_5)

    # Run 5x5 box filter on the image
    # with salt-and-pepper noise with
    # threshold = 0.05
    img_sp_05_box_5 = box_filter(img_sp_05, 5)
    cv2.imwrite('lena.sp.05.box.5x5.bmp', img_sp_05_box_5)

    # Run 5x5 box filter on the image
    # with salt-and-pepper noise with
    # threshold = 0.1
    img_sp_10_box_5 = box_filter(img_sp_10, 5)
    cv2.imwrite('lena.sp.10.box.5x5.bmp', img_sp_10_box_5)

    # Run 3x3 median filter on the image
    # with white Gaussian noise with
    # amplitude = 10
    img_gauss_10_med_3 = median_filter(img_gauss_10, 3)
    cv2.imwrite('lena.gaussian.10.median.3x3.bmp', img_gauss_10_med_3)

    # Run 3x3 median filter on the image
    # with white Gaussian noise with
    # amplitude = 30
    img_gauss_30_med_3 = median_filter(img_gauss_30, 3)
    cv2.imwrite('lena.gaussian.30.median.3x3.bmp', img_gauss_30_med_3)

    # Run 3x3 median filter on the image
    # with salt-and-pepper noise with
    # threshold = 0.05
    img_sp_05_med_3 = median_filter(img_sp_05, 3)
    cv2.imwrite('lena.sp.05.median.3x3.bmp', img_sp_05_med_3)

    # Run 3x3 median filter on the image
    # with salt-and-pepper noise with
    # threshold = 0.1
    img_sp_10_med_3 = median_filter(img_sp_10, 3)
    cv2.imwrite('lena.sp.10.median.3x3.bmp', img_sp_10_med_3)

    # Run 5x5 median filter on the image
    # with white Gaussian noise with
    # amplitude = 10
    img_gauss_10_med_5 = median_filter(img_gauss_10, 5)
    cv2.imwrite('lena.gaussian.10.median.5x5.bmp', img_gauss_10_med_5)

    # Run 5x5 median filter on the image
    # with white Gaussian noise with
    # amplitude = 30
    img_gauss_30_med_5 = median_filter(img_gauss_30, 5)
    cv2.imwrite('lena.gaussian.30.median.5x5.bmp', img_gauss_30_med_5)

    # Run 5x5 median filter on the image
    # with salt-and-pepper noise with
    # threshold = 0.05
    img_sp_05_med_5 = median_filter(img_sp_05, 5)
    cv2.imwrite('lena.sp.05.median.5x5.bmp', img_sp_05_med_5)

    # Run 5x5 median filter on the image
    # with salt-and-pepper noise with
    # threshold = 0.1
    img_sp_10_med_5 = median_filter(img_sp_10, 5)
    cv2.imwrite('lena.sp.10.median.5x5.bmp', img_sp_10_med_5)

    # Use octagon as kernel and set the orgin is at the center
    kernel = [
        [-2, -1], [-2, 0], [-2, 1],
        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
        [2, -1], [2, 0], [2, 1]
    ]

    # closing followed by opening on all noisy images
    img_gauss_10_close_open = opening(closing(img_gauss_10, kernel), kernel)
    img_gauss_30_close_open = opening(closing(img_gauss_30, kernel), kernel)
    img_sp_05_close_open = opening(closing(img_sp_05, kernel), kernel)
    img_sp_10_close_open = opening(closing(img_sp_10, kernel), kernel)

    cv2.imwrite('lena.gaussian.10.close.open.bmp', img_gauss_10_close_open)
    cv2.imwrite('lena.gaussian.30.close.open.bmp', img_gauss_30_close_open)
    cv2.imwrite('lena.sp.05.close.open.bmp', img_sp_05_close_open)
    cv2.imwrite('lena.sp.10.close.open.bmp', img_sp_10_close_open)

    # opening followed by closing on all noisy images
    img_gauss_10_open_close = closing(opening(img_gauss_10, kernel), kernel)
    img_gauss_30_open_close = closing(opening(img_gauss_30, kernel), kernel)
    img_sp_05_open_close = closing(opening(img_sp_05, kernel), kernel)
    img_sp_10_open_close = closing(opening(img_sp_10, kernel), kernel)

    cv2.imwrite('lena.gaussian.10.open.close.bmp', img_gauss_10_open_close)
    cv2.imwrite('lena.gaussian.30.open.close.bmp', img_gauss_30_open_close)
    cv2.imwrite('lena.sp.05.open.close.bmp', img_sp_05_open_close)
    cv2.imwrite('lena.sp.10.open.close.bmp', img_sp_10_open_close)


if __name__ == '__main__':
    main()
