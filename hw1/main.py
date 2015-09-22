import sys

import numpy as np
import cv2


def upside_down(img):
    img_ans = np.zeros(img.shape, np.int)
    for i in xrange(img.shape[0]):
        img_ans[i, :] = img[img.shape[0] - i - 1, :]
    return img_ans

def right_side_left(img):
    img_ans = np.zeros(img.shape, np.int)
    for j in xrange(img.shape[1]):
        img_ans[:, j] = img[:, img.shape[1] - j - 1]
    return img_ans

def diag_mirror(img):
    return right_side_left(upside_down(img))

def rotate(img):
    img_ans = np.zeros((img.shape[1], img.shape[0]), np.int)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            img_ans[j][img.shape[0] - i - 1] = img[i][j]
    return img_ans

def threshold(img, basic):
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if img[i][j] > 128:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img

def main():
    # usage: python ./main.py [task_id]
    # task_id = 1 ~ 6
    assert len(sys.argv) == 2
    assert int(sys.argv[1]) > 0 \
       and int(sys.argv[1]) < 7 \
       and int(sys.argv[1]) != 5

    img = cv2.imread('lena.bmp', 0)
    # img is a 512 x 512 np.array

    if sys.argv[1] == '1':
        #TODO: upside-down
        img_ans = upside_down(img)
        cv2.imwrite('1.upside_down.bmp', img_ans)

    elif sys.argv[1] == '2':
        # TODO: right-side-left
        img_ans = right_side_left(img)
        cv2.imwrite('2.right_side_left.bmp', img_ans)

    elif sys.argv[1] == '3':
        # TODO: diagonally mirrored
        img_ans = diag_mirror(img)
        cv2.imwrite('3.diag_mirror.bmp', img_ans)

    elif sys.argv[1] == '4':
        # TODO: rotate lena 45 degrees clockwise
        img_ans = rotate(img)
        cv2.imwrite('4.rotate.bmp', img_ans)

    else:
        # TODO: binarize lena at 128 to get a binary image
        img_ans = threshold(img, 128)
        cv2.imwrite('6.threshold_128.bmp', img)


if __name__ == '__main__':
    main()
