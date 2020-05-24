from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import greyscale_convert
from sobel_filter import sobel_filter_x, sobel_filter_y, gradient_magnitude


def non_max_suppression(img, img_theta):
    image_row, image_col = img.shape
    res = np.zeros(img.shape)
    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = img_theta[row, col]

            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = img[row, col - 1]
                after_pixel = img[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = img[row + 1, col - 1]
                after_pixel = img[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = img[row - 1, col]
                after_pixel = img[row + 1, col]

            else:
                before_pixel = img[row - 1, col - 1]
                after_pixel = img[row + 1, col + 1]

            if img[row, col] >= before_pixel and img[row, col] >= after_pixel:
                res[row, col] = img[row, col]
    return res


fig = plt.figure(figsize=(15, 5))
plt.gray()
img = greyscale_convert(Image.open("images/panther.jpg"))

a = fig.add_subplot(1, 4, 1)
plt.imshow(img)
a.set_title('Before')

a = fig.add_subplot(1, 4, 2)
img_blur = gaussian_filter(img, sigma=1)
plt.imshow(img_blur)
a.set_title('Gaussian filter')

img_sobel_x = sobel_filter_x(img_blur)
img_sobel_y = sobel_filter_y(img_blur)
img_gradient_magnitude = gradient_magnitude(img_sobel_x, img_sobel_y)
theta = np.arctan2(img_sobel_y, img_sobel_x)
a = fig.add_subplot(1, 4, 3)
plt.imshow(img_gradient_magnitude)
a.set_title('Gradient Magnitude')

a = fig.add_subplot(1, 4, 4)
img_non_mac_suppression = non_max_suppression(img_gradient_magnitude, theta)
plt.imshow(img_non_mac_suppression)
plt.title("Non Max Suppression")

plt.show()
