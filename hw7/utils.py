import numpy as np


def mark_interior_border(img_org):
    # define the operating function
    # for marking interior/border
    # pixel
    def h(c, d):
        if c == d:
            return c
        return 'b'
    img_ib = np.zeros(img_org.shape, np.int)
    # 0: background pixel
    # 1: interior pixel
    # 2: border pixel
    for i in xrange(img_org.shape[0]):
        for j in xrange(img_org.shape[1]):
            if img_org[i][j] > 0:
                # foreground pixel
                x1, x2, x3, x4 = 0, 0, 0, 0
                if i == 0:
                    if j == 0:
                        x1, x4 = img_org[i][j + 1], img_org[i + 1][j]
                    elif j == img_org.shape[1] - 1:
                        x3, x4 = img_org[i][j - 1], img_org[i + 1][j]
                    else:
                        x1, x3, x4 = img_org[i][j + 1], img_org[i][j - 1], img_org[i + 1][j]
                elif i == img_org.shape[0] - 1:
                    if j == 0:
                        x1, x2 = img_org[i][j + 1], img_org[i - 1][j]
                    elif j == img_org.shape[1] - 1:
                        x2, x3 = img_org[i - 1][j], img_org[i][j - 1]
                    else:
                        x1, x2, x3 = img_org[i][j + 1], img_org[i - 1][j], img_org[i][j - 1]
                else:
                    if j == 0:
                        x1, x2, x4 = img_org[i][j + 1], img_org[i - 1][j], img_org[i + 1][j]
                    elif j == img_org.shape[1] - 1:
                        x2, x3, x4 = img_org[i - 1][j], img_org[i][j - 1], img_org[i + 1][j]
                    else:
                        x1, x2, x3, x4 = img_org[i][j + 1], img_org[i - 1][j], img_org[i][j - 1], img_org[i + 1][j]
                x1 /= 255
                x2 /= 255
                x3 /= 255
                x4 /= 255
                a1 = h(1, x1)
                a2 = h(a1, x2)
                a3 = h(a2, x3)
                a4 = h(a3, x4)
                if a4 == 'b':
                    img_ib[i][j] = 2
                else:
                    img_ib[i][j] = 1
    return img_ib

def mark_pair_relationship(img_ib):
    # define the operating function
    # for marking pair relationship
    def h(a, m):
        if a == m:
            return 1
        return 0
    # background pixel: 0
    # p: 1
    # q: 2
    img_pair = np.zeros(img_ib.shape, np.int)
    for i in xrange(img_ib.shape[0]):
        for j in xrange(img_ib.shape[1]):
            if img_ib[i][j] > 0:
                # foreground pixel,
                # including interior and border
                x1, x2, x3, x4 = 0, 0, 0, 0
                if i == 0:
                    if j == 0:
                        x1, x4 = img_ib[i][j + 1], img_ib[i + 1][j]
                    elif j == img_ib.shape[1] - 1:
                        x3, x4 = img_ib[i][j - 1], img_ib[i + 1][j]
                    else:
                        x1, x3, x4 = img_ib[i][j + 1], img_ib[i][j - 1], img_ib[i + 1][j]
                elif i == img_ib.shape[0] - 1:
                    if j == 0:
                        x1, x2 = img_ib[i][j + 1], img_ib[i - 1][j]
                    elif j == img_ib.shape[1] - 1:
                        x2, x3 = img_ib[i - 1][j], img_ib[i][j - 1]
                    else:
                        x1, x2, x3 = img_ib[i][j + 1], img_ib[i - 1][j], img_ib[i][j - 1]
                else:
                    if j == 0:
                        x1, x2, x4 = img_ib[i][j + 1], img_ib[i - 1][j], img_ib[i + 1][j]
                    elif j == img_ib.shape[1] - 1:
                        x2, x3, x4 = img_ib[i - 1][j], img_ib[i][j - 1], img_ib[i + 1][j]
                    else:
                        x1, x2, x3, x4 = img_ib[i][j + 1], img_ib[i - 1][j], img_ib[i][j - 1], img_ib[i + 1][j]
                if h(x1, 1) + h(x2, 1) + h(x3, 1) + h(x4, 1) >= 1 and   \
                   img_ib[i][j] == 2:
                    img_pair[i][j] = 1
                else:
                    img_pair[i][j] = 2
    return img_pair

def compute_yokoi_number(img_org):
    # define the operation function for
    # Yokoi Connectivity Number ...
    def h(b, c, d, e):
        if b == c and (d != b or e != b):
            return 'q'
        if b == c and (d == b and e == b):
            return 'r'
        return 's'

    yokoi_map = np.zeros(img_org.shape, np.int)
    # compute and output Yokoi Connectivity Number ...
    for i in xrange(img_org.shape[0]):
        for j in xrange(img_org.shape[1]):
            if img_org[i][j] > 0:  # foreground pixel
                if i == 0:
                    if j == 0:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = 0, img_org[i + 1][j], img_org[i + 1][j + 1]
                    elif j == img_org.shape[1] - 1:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], 0
                        x8, x4, x5 = img_org[i + 1][j - 1], img_org[i + 1][j], 0
                    else:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = img_org[i + 1][j - 1], img_org[i + 1][j], img_org[i + 1][j + 1]
                elif i == img_org.shape[0] - 1:
                    if j == 0:
                        x7, x2, x6 = 0, img_org[i - 1][j], img_org[i - 1][j + 1]
                        x3, x0, x1 = 0, img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                    elif j == img_org.shape[1] - 1:
                        x7, x2, x6 = img_org[i - 1][j - 1], img_org[i - 1][j], 0
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], 0
                        x8, x4, x5 = 0, 0, 0
                    else:
                        x7, x2, x6 = img_org[i - 1][j - 1], img_org[i - 1][j], img_org[i - 1][j + 1]
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                else:
                    if j == 0:
                        x7, x2, x6 = 0, img_org[i - 1][j], img_org[i - 1][j + 1]
                        x3, x0, x1 = 0, img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = 0, img_org[i + 1][j], img_org[i + 1][j + 1]
                    elif j == img_org.shape[1] - 1:
                        x7, x2, x6 = img_org[i - 1][j - 1], img_org[i - 1][j], 0
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], 0
                        x8, x4, x5 = img_org[i + 1][j - 1], img_org[i + 1][j], 0
                    else:
                        x7, x2, x6 = img_org[i - 1][j - 1], img_org[i - 1][j], img_org[i - 1][j + 1]
                        x3, x0, x1 = img_org[i][j - 1], img_org[i][j], img_org[i][j + 1]
                        x8, x4, x5 = img_org[i + 1][j - 1], img_org[i + 1][j], img_org[i + 1][j + 1]

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
                yokoi_map[i][j] = ans
    return yokoi_map

def main():
    # TODO: test with written functions ...
    img_org = np.array(
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]
        ]
    )
    img_ib = mark_interior_border(img_org * 255)
    print img_ib

    img_ib = np.array(
        [[2, 2, 2, 2, 2, 2, 0],
         [2, 1, 1, 1, 1, 2, 0],
         [2, 2, 2, 2, 1, 2, 2],
         [2, 0, 0, 2, 2, 1, 2],
         [2, 0, 0, 0, 2, 2, 2],
         [2, 0, 0, 0, 0, 0, 2]
        ]
    )

    img_pair = mark_pair_relationship(img_ib)
    print img_pair


if __name__ == '__main__':
    main()
