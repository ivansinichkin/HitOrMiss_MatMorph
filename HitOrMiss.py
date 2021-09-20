import PIL
import PIL.Image as Image
import PIL.ImageOps
import matplotlib.pylab as plt
import numpy as np
import cv2


def gener(p, q, neven, meven):
    for i in range(-p, p + 1 + neven):
        for j in range(-q, q + 1 + meven):
            yield i, j


def my_erosion(img, struct):
    out = np.zeros(img.shape, dtype='int')
    H = (struct.shape[0] - 1) // 2
    W = (struct.shape[1] - 1) // 2
    '''для прохода по всем значениям struct при ее четных размерностях
    необходимы значения генератора gener условно от -p до p + 2
    для этого создадим флаги четности размерности Neven и Meven
    при четном значении размерности struct флаг примет значение 1
    это значение будет добавлено к верхней границе значений,
    выдаваемых gener'''
    Neven = (struct.shape[0] - 1) % 2
    Meven = (struct.shape[1] - 1) % 2
    for i in range(H, img.shape[0] - H):
        for j in range(W, img.shape[1] - W):
            for m, n in gener(H, W, Neven, Meven):
                if not struct[H + m, W + n]:
                    continue
                if struct[H + m, W + n] != img[i + m - Neven, j + n - Meven]:
                    break
            else:
                out[i, j] = 255
    return out.astype(img.dtype)


def my_intersection(a, b):
    out = np.zeros(a.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            sum = int(a[i, j]) + int(b[i, j])
            if sum != 510:
                out[i, j] = 0
            else:
                out[i, j] = 255
    return out


def get_localback(struct):
    localback = np.pad(struct, (1, 1), 'constant')
    localback = np.where(localback == 0, 255, 0)
    return localback


def get_complement(img):
    complement = np.where(img == 0, 255, 0)
    return complement


def hit_or_miss(img, struct):
    localback = get_localback(struct)
    complement = get_complement(img)
    comp_er_lb = my_erosion(complement, localback)
    img_er_struct = my_erosion(img, struct)
    h_o_m = my_intersection(comp_er_lb, img_er_struct)
    result = np.argwhere(h_o_m > 1)
    return h_o_m, result


'''img = cv2.imread('HoM.png', cv2.IMREAD_GRAYSCALE)
struct = cv2.imread('K.png', cv2.IMREAD_GRAYSCALE)
print(struct.shape)
picture, data = hit_or_miss(img, struct)
plt.imshow(picture, cmap='gray')
plt.show()
print(data)'''
