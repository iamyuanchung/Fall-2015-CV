import sys

import numpy as np
import cv2


def convolve(A, B):
    assert A.shape == B.shape
    value = 0
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            value += (A[i, j] * B[B.shape[0] - i - 1, B.shape[1] - j - 1])
    return value

def convolve_img(img, kernel):
    img_new = np.zeros((img.shape[0] - kernel.shape[0] + 1, img.shape[1] - kernel.shape[1] + 1))
    for i in xrange(img_new.shape[0]):
        for j in xrange(img_new.shape[1]):
            img_new[i, j] = convolve(img[i:i+kernel.shape[0], j:j+kernel.shape[1]], kernel)
    return img_new

def main():
    """
        >> python ./main.py [task_id] [threshold]

        [task_id]
            1: Laplace Mask Type-1
            2: Laplace Mask Type-2
            3: Minimum variance Laplacian
            4: Laplace of Gaussian
            5: Difference of Gaussian
    """
    assert len(sys.argv) == 3
    task_id = sys.argv[1]
    assert int(task_id) >= 1 and int(task_id) <= 5
    threshold = sys.argv[2]

    img = cv2.imread('lena.bmp', 0)

    if task_id == '1':
        print 'Laplace Mask Type-1, thresholds at ' +  threshold
        print ''
        k = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        img_ans = convolve_img(img, k)
        cv2.imwrite('lena.laplacian.type1.' + threshold + '.bmp', (img_ans < int(threshold)) * 255)

    elif task_id == '2':
        print 'Laplace Mask Type-2, thresholds at ' +  threshold
        print ''
        k = np.array([
            [1., 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ]) / 3
        img_ans = convolve_img(img, k)
        cv2.imwrite('lena.laplacian.type2.' + threshold + '.bmp', (img_ans < int(threshold)) * 255)

    elif task_id == '3':
        print 'Minimum variance Laplacian, thresholds at ' +  threshold
        print ''
        k = np.array([
            [2., -1, 2],
            [-1, -4, -1],
            [2, -1, 2]
        ]) / 3
        img_ans = convolve_img(img, k)
        cv2.imwrite('lena.min.var.laplacian.' + threshold + '.bmp', (img_ans < int(threshold)) * 255)

    elif task_id == '4':
        print 'Laplace of Gaussian, thresholds at ' +  threshold
        print ''
        k = np.array([
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
        ])
        img_ans = convolve_img(img, k)
        cv2.imwrite('lena.laplacian.of.gaussian.' + threshold + '.bmp', (img_ans < int(threshold)) * 255)

    else:
        print 'Difference of Gaussian, thresholds at ' +  threshold
        print ''
        k = np.array([
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        ])
        img_ans = convolve_img(img, k)
        cv2.imwrite('lena.difference.of.gaussian.' + threshold + '.bmp', (img_ans >= int(threshold)) * 255)


if __name__ == '__main__':
    main()
