#!/usr/bin/env python


# do we include test images?
# yeah, because they're being concatenated, so we can
# always create test/train data sets from this!!!

# we *assume* that we can import tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import uuid

# we load the MNIST database ...
# ... and cache it locally for efficent access
from tensorflow.examples.tutorials.mnist import input_data
# include for any debuggin purposes
tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# just in case ...
mnist_dim=28

def filter_mnist_by_digit(labels, filterdigit):
    """(np.array, int) -> [int]
    Return the indices corresponding to the value of filterdigit in the
    provided labels.

    :param labels: numpy array of shape (x, 10), each row containing a
    one-hot-encoded vector corresponding to the digit value of the associated
    mnist image. Assumes the encoding is zero-indexed.
    :param filterdigit: int corresponding to the digit to be filtered.
    """
    return [idx for idx,row in enumerate(labels) if row[filterdigit]==1]



def get_mnist_images_digit(digit):
    """
    Return all of the mnist examples of the specified digit from MNIST,
    across the train, validation, and test sets.
    :param digit: the digit to retrieve
    """
    idx_train = filter_mnist_by_digit(mnist.train.labels, digit)
    train = mnist.train.images[idx_train]
    idx_test = filter_mnist_by_digit(mnist.test.labels, digit)
    test = mnist.test.images[idx_test]
    idx_val = filter_mnist_by_digit(mnist.validation.labels, digit)
    val = mnist.validation.images[idx_val]
    return np.vstack([train, test, val])

# Randomly take an image from the test set
def get_random_image_index(imgs):
    """(tensorflow.contrib.learn.python.learn.datasets.base.Datasets) -> int
        Return all of the mnist examples of the specified digit from tf_mnist,
        across the train, validation, and test sets.
        :param imgs: labels corresponding to the MNIST images
        """
    return np.random.randint(0, len(imgs) - 1)


def get_zero_matrix(width, height=mnist_dim):
    """
    Get a zero matrix (2D array) of requested dimensions.
    :param width: width of zero matrix
    :param height: height of zero matrix, defaults to MNIST's 28
    :return: np.zeros array of height, width
    """
    return np.zeros((height, width))

def get_random_spaces(_min, _max, length):
    """
    Get a list of 'length' zero matrices with randomly determined width.
    :param _min: left side of the random interval
    :param _max: right side of the interval
    :param length: number of 2D arrays to generate
    :return: list of 2D np.array's with randomly generated width
    """
    # ensures that we have a list with width at least one
    # assert _max > _min
    return [get_zero_matrix(w) for w in get_uniform_integer_sequence(_min, _max, length)]



def get_uniform_integer_sequence(low, high, size):
    """
    Return an array of random numbers of the given size
    where the number x is such that low <- x <- high
    :param low: lower bound
    :param high: upper bound (exclusive)
    :param size: length of req'd array
    :return: an array of random integers between the spec'd numbers
    """
    return np.random.randint(low, high, size)


def shrink_or_grow_image(width, _default=mnist_dim):
    """
    Taking the user-defined width, do we need to shrink or
    grow the images?
    :param width: user-supplied requested width of the image
    :param _default: the default dimension of MNIST images
    :return: Booleans indicating whether to shrink or grow the image
    """
    spacing = _default - width
    # we need to shrink the image
    if spacing > 0:
        return True, False
    # we need to grow the image
    elif spacing < 0:
        return False, True

    return False, False

def retrieve_images(digits):
    """
    Gets a list of random images from the MNIST database
    corresponding to each of the numbers in digits
    :param digits: list of digits to get from MNIST
    :return: A randomly selected list of MNIST images
    """
    initial_mnist_images=[]

    for digit in digits:
        imgs = get_mnist_images_digit(digit)
        idx = get_random_image_index(imgs)
        initial_mnist_images.append(np.reshape(imgs[idx, :], (mnist_dim, mnist_dim)))

    return initial_mnist_images

def shrink_matrix(m, cut):
    """
    Cut down the width of images matrices to the given size
    :param m: matrix (2D np.array)
    :param cut: number of columns to cut from the matrix
    :return: matrix with given number of columns removed
    """
    end = mnist_dim
    start = end - cut

    l = [i for i in range(start, end)]

    return np.delete(m, l, 1)

def grow_matrix(m, paste):
    """
    Create a copy of the MNIST image with added, blank columns
    :param m: matrix (2D np.array)
    :param paste: number of columns to append to matrix
    :return: matrix with given number of columns appended
    """
    h = m.shape[0]
    zeroes = np.zeros((h, paste))
    return np.hstack((m, zeroes))

def concat_matrix_lists(l1, l2):
    """
    Produce a list of matrices zipped together
    :param l1: list of matrices
    :param l2: list of spacing (zero value) matrices
    :return: both lists zipped alternately together
    """
    last_idx = len(l1) - 1
    l = []

    for digits, space in list(zip(l1, l2)):
        l.append(digits)
        l.append(space)

    l.append(l1[last_idx])

    return l

def check_input_numbers(digits):
    """
    Check whether or not the list of digits is in the correct interval [0-9].
    :param digits: list of digits to check
    :return: True if the numbers are in the required range, else False
    """
    if all(0 <= number < 10 for number in digits):
        return True
    return False


def check_input_type(digits):
    """
    Check whether the input list is composed of integers.
    :param digits: list to be tested
    :return:
    """
    for digit in digits:
        try:
            int(digit)
        except ValueError:
            return False
    return True


def check_spacing_range(spacing):
    """
    Check the spacing_range parameter.
    :param spacing: tuple to be tested
    :return: True if the spacing range is valid, else False
    """
    _min, _max = spacing[0], spacing[1]
    if not(check_input_type([_min, _max])):
        return False
    if _max < _min:
        return False
    return True


def check_width(width):
    """
    See whether the 'width' parameter is valid
    :param width: user input width parameter
    :return: if the width param passes the tests, True, else False
    """
    # check the type of the parameter
    if not(check_input_type([width])):
        return False
    if width <= 0:
        return False
    return True

def generate_numbers_sequence(digits, spacing_range, image_width):
    """
    Generate an image that contains the sequence of given numbers, spaced
    randomly using an uniform distribution.

    Parameters
    ----------
    digits:
	A list-like containing the numerical values of the digits from which
        the sequence will be generated (for example [3, 5, 0]).
    spacing_range:
	a (minimum, maximum) pair (tuple), representing the min and max spacing
        between digits. Unit should be pixel.
    image_width:
        specifies the width of the image in pixels.

    Returns
    -------
    The image containing the sequence of numbers. Images should be represented
    as floating point 32bits numpy arrays with a scale ranging from 0 (black) to
    1 (white), the first dimension corresponding to the height and the second
    dimension to the width.
    :param spacing_range:
    :param image_width:
    """

    ## TODO: check the params for any obvious nonsense or frivolity

    if not(check_input_type(digits)) or not(check_input_numbers(digits)) or not(digits):
        raise Exception("error: input list 'digits' has incorrect values")

    if not(check_spacing_range(spacing_range)):
        raise Exception("error: input tuple 'spacing_range' has incorrect values")

    if not(check_width(image_width)):
        raise Exception("error: input tuple 'image_width' has incorrect values")

    mnist_images         = []
    initial_mnist_images = retrieve_images(digits)

    # what width have we been passed? do we shrink, or
    # embiggen, the image?
    shrink, grow = shrink_or_grow_image(image_width)

    if shrink:
        shrinkage = mnist_dim - image_width
        mnist_images = [shrink_matrix(_img, shrinkage) for _img in initial_mnist_images]


    if grow:
        growth = image_width - mnist_dim
        mnist_images = [grow_matrix(_img, growth) for _img in initial_mnist_images]

    if not grow and not shrink:
        mnist_images = initial_mnist_images

    ## TODO: else, copy initial_mnist_images to mnist_images

    # get our min, max parameters for the random spacing
    low =  min(spacing_range)
    high = max(spacing_range)

    # generate our spacing between the selected digits
    num_of_spacings = len(digits) - 1
    spacings = get_random_spaces(low, high, num_of_spacings)

    # interweave our matrices
    concat_image_list = concat_matrix_lists(mnist_images, spacings)

    # create our munged single image
    concat_image = np.concatenate(concat_image_list, axis=1)

    ## TODO: JKK: got to ensure that this doesn't plot the digit!!!
    ## TODO: could cause problems!!!
    unique_filename = 'mnist_' + str(uuid.uuid4()) + '.png'
    # Turn interactive plotting off
    plt.ioff()
    img = plt.imshow(concat_image)
    img.set_cmap('gray')
    plt.axis('off')
    plt.savefig(unique_filename, bbox_inches='tight')

    return concat_image
