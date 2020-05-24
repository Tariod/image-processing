import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from utils import greyscale_convert


def sobel_filter_x(image):
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return convolve(image, g_x)


def sobel_filter_y(image):
    g_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return convolve(image, g_y)


def gradient_magnitude(img_x, img_y):
    magnitude = np.sqrt(np.square(img_x.copy()) + np.square(img_y.copy()))
    magnitude *= 255 / magnitude.max()
    return magnitude.astype(int)


fig = plt.figure(figsize=(15, 5))
img = Image.open("images/panther.jpg")

a = fig.add_subplot(1, 5, 1)
plt.imshow(img)
a.set_title('Before')

a = fig.add_subplot(1, 5, 2)
img_greyscale = greyscale_convert(img)
plt.imshow(img_greyscale, cmap='gray')
a.set_title('Greyscale')

a = fig.add_subplot(1, 5, 3)
img_sobel_x = sobel_filter_x(img_greyscale)
plt.imshow(img_sobel_x, cmap='gray')
a.set_title('Sobel X')

a = fig.add_subplot(1, 5, 4)
img_sobel_y = sobel_filter_y(img_greyscale)
plt.imshow(img_sobel_y, cmap='gray')
a.set_title('Sobel Y')

a = fig.add_subplot(1, 5, 5)
img_sobel_combined = gradient_magnitude(img_sobel_x, img_sobel_y)
plt.imshow(img_sobel_combined, cmap='gray')
a.set_title('Gradient Magnitude')

plt.show()
