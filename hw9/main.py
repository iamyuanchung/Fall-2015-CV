import sys

import numpy as np
import cv2


def border_extend(img):
    img_ext = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    img_ext[0, 1:-1] = img[0, :]    # copy the first row
    img_ext[-1, 1:-1] = img[-1, :]  # copy the last row
    img_ext[1:-1, 0] = img[:, 0]    # copy the first column
    img_ext[1:-1, -1] = img[:, -1]  # copy the last column
    img_ext[0, 0] = img[0, 0]       # copy the top-left element
    img_ext[0, -1] = img[0, -1]     # copy the top-right element
    img_ext[-1, 0] = img[-1, 0]     # copy the bottom-left element
    img_ext[-1, -1] = img[-1, -1]   # copy the bottom-right element
    img_ext[1:-1, 1:-1] = img[:, :] # copy the rest
    return img_ext

def convolve(A, B):
    assert A.shape == B.shape
    value = 0
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            value += (A[i, j] * B[B.shape[0] - i - 1, B.shape[1] - j - 1])
    return value

def Roberts(img):
    img = np.asarray(a=img, dtype=np.int)
    k1 = np.array([
        [1, 0],
        [0, -1]
    ])
    k2 = np.array([
        [0, 1],
        [-1, 0]
    ])
    Gx = np.zeros((img.shape[0] - 1, img.shape[1] - 1), np.int)
    Gy = np.zeros((img.shape[0] - 1, img.shape[1] - 1), np.int)
    for i in xrange(Gx.shape[0]):
        for j in xrange(Gx.shape[1]):
            Gx[i, j] = convolve(img[i:i+2, j:j+2], k1)
            Gy[i, j] = convolve(img[i:i+2, j:j+2], k2)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    return G

def Prewitt(img):
    img = np.asarray(a=img, dtype=np.int)
    k1 = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    k2 = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    Gx = np.zeros((img.shape[0] - 2, img.shape[1] - 2), np.int)
    Gy = np.zeros((img.shape[0] - 2, img.shape[1] - 2), np.int)
    for i in xrange(Gx.shape[0]):
        for j in xrange(Gx.shape[1]):
            Gx[i, j] = convolve(img[i:i+3, j:j+3], k1)
            Gy[i, j] = convolve(img[i:i+3, j:j+3], k2)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    return G

def Sobel(img):
    img = np.asarray(a=img, dtype=np.int)
    k1 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    k2 = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    Gx = np.zeros((img.shape[0] - 2, img.shape[1] - 2), np.int)
    Gy = np.zeros((img.shape[0] - 2, img.shape[1] - 2), np.int)
    for i in xrange(Gx.shape[0]):
        for j in xrange(Gx.shape[1]):
            Gx[i, j] = convolve(img[i:i+3, j:j+3], k1)
            Gy[i, j] = convolve(img[i:i+3, j:j+3], k2)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    return G

def Frei(img):
    img = np.asarray(a=img, dtype=np.int)
    k1 = np.array([
        [-1, -np.sqrt(2), -1],
        [0, 0, 0],
        [1, np.sqrt(2), 1]
    ])
    k2 = np.array([
        [-1, 0, 1],
        [-np.sqrt(2), 0, np.sqrt(2)],
        [-1, 0, 1]
    ])
    Gx = np.zeros((img.shape[0] - 2, img.shape[1] - 2), np.int)
    Gy = np.zeros((img.shape[0] - 2, img.shape[1] - 2), np.int)
    for i in xrange(Gx.shape[0]):
        for j in xrange(Gx.shape[1]):
            Gx[i, j] = convolve(img[i:i+3, j:j+3], k1)
            Gy[i, j] = convolve(img[i:i+3, j:j+3], k2)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    return G
    

def Kirsch(img):
    img = np.asarray(a=img, dtype=np.int)
    k0 = np.array([
        [-3, -3, 5],
        [-3, 0, 5],
        [-3, -3, 5]
    ])
    k1 = np.array([
        [-3, 5, 5],
        [-3, 0, 5],
        [-3, -3, -3]
    ])
    k2 = np.array([
        [5, 5, 5],
        [-3, 0, -3],
        [-3, -3, -3]
    ])
    k3 = np.array([
        [5, 5, -3],
        [5, 0, -3],
        [-3, -3, -3]
    ])
    k4 = np.array([
        [5, -3, -3],
        [5, 0, -3],
        [5, -3, -3]
    ])
    k5 = np.array([
        [-3, -3, -3],
        [5, 0, -3],
        [5, 5, -3]
    ])
    k6 = np.array([
        [-3, -3, -3],
        [-3, 0, -3],
        [5, 5, 5]
    ])
    k7 = np.array([
        [-3, -3, -3],
        [-3, 0, 5],
        [-3, 5, 5]
    ])
    G = np.zeros((img.shape[0] - 2, img.shape[1] - 2), np.int)
    for i in xrange(G.shape[0]):
        for j in xrange(G.shape[1]):
            g0 = convolve(img[i:i+3, j:j+3], k0)
            g1 = convolve(img[i:i+3, j:j+3], k1)
            g2 = convolve(img[i:i+3, j:j+3], k2)
            g3 = convolve(img[i:i+3, j:j+3], k3)
            g4 = convolve(img[i:i+3, j:j+3], k4)
            g5 = convolve(img[i:i+3, j:j+3], k5)
            g6 = convolve(img[i:i+3, j:j+3], k6)
            g7 = convolve(img[i:i+3, j:j+3], k7)
            G[i, j] = np.max([g0, g1, g2, g3, g4, g5, g6, g7])
    return G

def Robinson(img):
    img = np.asarray(a=img, dtype=np.int)
    G = np.zeros((img.shape[0] - 2, img.shape[1] - 2), np.int)
    k0 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    k1 = np.array([
        [0, 1, 2],
        [-1, 0, 1],
        [-2, -1, 0]
    ])
    k2 = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    k3 = np.array([
        [2, 1, 0],
        [1, 0, -1],
        [0, -1, -2]
    ])
    k4 = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    k5 = np.array([
        [0, -1, -2],
        [1, 0, -1],
        [2, 1, 0]
    ])
    k6 = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    k7 = np.array([
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2]
    ])
    for i in xrange(G.shape[0]):
        for j in xrange(G.shape[1]):
            r0 = convolve(img[i:i+3, j:j+3], k0)
            r1 = convolve(img[i:i+3, j:j+3], k1)
            r2 = convolve(img[i:i+3, j:j+3], k2)
            r3 = convolve(img[i:i+3, j:j+3], k3)
            r4 = convolve(img[i:i+3, j:j+3], k4)
            r5 = convolve(img[i:i+3, j:j+3], k5)
            r6 = convolve(img[i:i+3, j:j+3], k6)
            r7 = convolve(img[i:i+3, j:j+3], k7)
            G[i, j] = np.max([r0, r1, r2, r3, r4, r5, r6, r7])
    return G

def Nevatia(img):
    img = np.asarray(a=img, dtype=np.int)
    k0 = np.array([
        [100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100],
        [0, 0, 0, 0, 0],
        [-100, -100, -100, -100, -100],
        [-100, -100, -100, -100, -100],
    ])
    k1 = np.array([
        [100, 100, 100, 100, 100],
        [100, 100, 100, 78, -32],
        [100, 92, 0, -92, -100],
        [32, -78, -100, -100, -100],
        [-100, -100, -100, -100, -100]
    ])
    k2 = np.array([
        [100, 100, 100, 32, -100],
        [100, 100, 92, -78, -100],
        [100, 100, 0, -100, -100],
        [100, 78, -92, -100, -100],
        [100, -32, -100, -100, -100]
    ])
    k3 = np.array([
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100]
    ])
    k4 = np.array([
        [-100, 32, 100, 100, 100],
        [-100, -78, 92, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, -92, 78, 100],
        [-100, -100, -100, -32, 100]
    ])
    k5 = np.array([
        [100, 100, 100, 100, 100],
        [-32, 78, 100, 100, 100],
        [-100, -92, 0, 92, 100],
        [-100, -100, -100, -78, 32],
        [-100, -100, -100, -100, -100]
    ])
    G = np.zeros((img.shape[0] - 4, img.shape[1] - 4), np.int)
    for i in xrange(G.shape[0]):
        for j in xrange(G.shape[1]):
            n0 = np.sum(img[i: i + 5, j: j + 5] * k0)
            n1 = np.sum(img[i: i + 5, j: j + 5] * k1)
            n2 = np.sum(img[i: i + 5, j: j + 5] * k2)
            n3 = np.sum(img[i: i + 5, j: j + 5] * k3)
            n4 = np.sum(img[i: i + 5, j: j + 5] * k4)
            n5 = np.sum(img[i: i + 5, j: j + 5] * k5)
            G[i, j] = np.max([n0, n1, n2, n3, n4, n5])
    return G

def main():
    """
        >> python ./main.py [operator] [threshold]

    """
    assert len(sys.argv) == 3

    operator = sys.argv[1]
    threshold = sys.argv[2]
    assert operator == 'roberts' or operator == 'prewitt'   \
        or operator == 'sobel' or operator == 'frei'        \
        or operator == 'kirsch' or operator == 'robinson'   \
        or operator == 'nevatia'

    img = cv2.imread('lena.bmp', 0)

    print 'Operator: ' + operator
    print 'Threshold: ' + threshold
    print ''

    if operator == 'roberts':
        img_roberts = (Roberts(img) <= int(threshold)) * 255
        cv2.imwrite('lena.roberts.' + threshold + '.bmp', img_roberts)

    elif operator == 'prewitt':
        img_prewitt = (Prewitt(img) <= int(threshold)) * 255
        cv2.imwrite('lena.prewitt.' + threshold + '.bmp', img_prewitt)

    elif operator == 'sobel':
        img_sobel = (Sobel(img) <= int(threshold)) * 255
        cv2.imwrite('lena.sobel.' + threshold + '.bmp', img_sobel)

    elif operator == 'frei':
        img_frei = (Frei(img) <= int(threshold)) * 255
        cv2.imwrite('lena.frei.' + threshold + '.bmp', img_frei)

    elif operator == 'kirsch':
        img_kirsch = (Kirsch(img) <= int(threshold)) * 255
        cv2.imwrite('lena.kirsch.' + threshold + '.bmp', img_kirsch)

    elif operator == 'robinson':
        img_robinson = (Robinson(img) <= int(threshold)) * 255
        cv2.imwrite('lena.robinson.' + threshold + '.bmp', img_robinson)

    else:
        img_nevatia = (Nevatia(img) <= int(threshold)) * 255
        cv2.imwrite('lena.nevatia.' + threshold + '.bmp', img_nevatia)


if __name__ == '__main__':
    main()
