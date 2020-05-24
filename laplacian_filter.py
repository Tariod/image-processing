import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from utils import greyscale_convert


def laplacian_filter_negative(image):
    g_x = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return convolve(image, g_x)


def laplacian_filter_positive(image):
    g_y = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve(image, g_y)


fig = plt.figure(figsize=(15, 5))
img = Image.open("images/panther.jpg")

a = fig.add_subplot(1, 4, 1)
plt.imshow(img)
a.set_title('Before')

a = fig.add_subplot(1, 4, 2)
img_greyscale = greyscale_convert(img)
plt.imshow(img_greyscale, cmap='gray')
a.set_title('Greyscale')

a = fig.add_subplot(1, 4, 3)
img_laplacian_negative = laplacian_filter_negative(img_greyscale)
plt.imshow(img_laplacian_negative, cmap='gray')
a.set_title('Laplacian Negative')

a = fig.add_subplot(1, 4, 4)
img_laplacian_positive = laplacian_filter_positive(img_greyscale)
plt.imshow(img_laplacian_positive, cmap='gray')
a.set_title('Laplacian Positive')

plt.show()
