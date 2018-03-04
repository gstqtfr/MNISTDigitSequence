#-*- coding: utf-8 -*-

import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
from skimage import io, transform, util

def downscale_by_mean(img, cval=0.0):
    """
    Down-samples image by local averaging.
    :param img: image to downsample
    :param cval: value to use for padding the image, if necessary.
    :return: down-sampled image, input image dimensions preserved.
    """
    pass

def edge_detection(img):
    """
    Applies the Canny edge detection algorithm, and produces
    a very stark image. Conuterintuitively, may confuse nets
    trained on less images with less contrast.
    :param img: image to perform edge detection on
    :return: high contrast image with little detail
    """
    # uses the Canny edge detector: https://en.wikipedia.org/wiki/Canny_edge_detector
    from skimage.feature import canny
    edges = canny(img / 1.)
    from scipy import ndimage as ndi
    return ndi.binary_fill_holes(edges)


def rotate_image(img, rotation):
    """
    Rotates the image by the given offset, using a scaled rotation with
    adjustable centre of rotation so that you can rotate at any location you prefer.
    :param img: image to be rotated
    :param rotation: number of degrees to rotate the image
    :return: rotated image
    """
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    return cv2.warpAffine(img, M, (cols, rows))

def dilate_image(img, kernel_size, iterations = 1):
    """
    Performas a 'dilation' of the image. Here, a pixel element is ‘1’ if at least one
    pixel under the kernel is ‘1’. It increases the white region in the image or size
    of the foreground object increases.
    :param img: image to dilate
    :param kernel_size: size of kernel: 5 give you a 5x5 kernel, etc.
    :param iterations: number of times to run the dilation, defaults to 1

    :return: dilated image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)

def black_hat_morph(img, kernel_size):
    """
    Apply the Black Hat  transform to the image.
    :param img: input inage to be  orphed
    :param kernel_size: size of the kernel to use for morphing
    :return: morphed image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def filter_2D_conv(img, kernel_size):
    """
    Convolve a kernel with an image. We apply an averaging filter on an image.
    :param img: image to be convolved.
    :param kernel_size: size of our convolution filter
    :return: convolved image
    """
    coeff = kernel_size * kernel_size
    kernel = np.ones((kernel_size, kernel_size), np.float32) / coeff
    return cv2.filter2D(img, -1, kernel)

def median_filter(img, factor=5):
    """
    Computes the median of all the pixels under the kernel window; the
    central pixel is replaced with this median value.
    :param img: image to transform
    :param factor: degree by which to filter
    :return: filtered image
    """
    return cv2.medianBlur(img, factor)

def add_gaussian_noise(img):
    """
    Adds Gaussian noise to the image.
    :param img: input to be processed
    :return: imagewith added Gaussian noise
    """

    return util.random_noise(img, mode='gaussian')




