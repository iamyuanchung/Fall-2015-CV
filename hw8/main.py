import copy

import numpy as np
import cv2


def generate_Gaussian_noise(img, mu, sigma, amplitude):
    return img + amplitude * np.random.normal(mu, sigma, img.shape)

def generate_salt_and_pepper_noise(img, low, high, threshold):
    prob_map = np.random.uniform(low, high, img.shape)
    img_sp = copy.deepcopy(img)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if prob_map[i][j] < threshold:
                img_sp[i][j] = 0
            elif prob_map[i][j] > 1 - threshold:
                img_sp[i][j] = 255
    return img_sp


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

    


if __name__ == '__main__':
    main()
