# MNISTDigitSequence

This is a small project intended to provide images of sequences of numbers, drawn from the 
[MNIST data set](http://yann.lecun.com/exdb/mnist/). The purpose of the project is to provide
data for deep learning networks, such as classifiers or generative models, by augmenting
the MNIST data set.

## Software versions

The software was developed in Python 3 (3.6.4). 

Libraries used are:

- Tensorflow 1.5.0
- Numpy 1.14.0
- Scikit-Image 0.13.1
- OpenCV 3.3.1

An effort has been made to minimise use of external Python libraries.

## Functional specification

The software was designed to meet the following specifications:

- select a _sequence_ of MNIST digits 
- generate a spacing to separate the digits, chosen from a uniform random distribution
- crop, or grow, the digits in accordance with a user-defined width parameter
- combine the digits, and the spacings between them, into a single image

The intention is to provide an API/command-line module which takes as parameters:

- digits: the list of numbers to select from MNIST
- spacing_range: a minimum, maximum pair used to provide an interval for the generated spacings
- image_width: used to either crop or widen the images

The main function to call is the following:

```python
def generate_numbers_sequence(digits, spacing_range, image_width):
    """
    Generate an image that contains the sequence of given numbers, spaced
    randomly using a uniform distribution.
    """
```

So, for example, if we had

- digits = [0,0,5,7,1]
- spacing_range = (3,10)
- width = 27

then 
```python
generate_numbers_sequence(digits, spacing_range, image_width)
```

would read through and randomly select MNIST images corresponding to each of the numbers 
provided, create arrays of spacings between each of the digits, and concatenate them into 
a single np.array. Each MNIST image would be either cropped or grown, according to the width 
parameter; in this case, image_width = 27, so each MNIST image would be cropped by 1 pixel 
(or, rather, one column of the np.array). 

The function would return a floating point 32-bit numpy array representing the sequence of
digits. Each float in the array represents a colour value for the pixel, such that 1 == 
white and 0 == black.

In addition, the generated image is saved in the current directory as a .png.

## Unit testing

Tests for the Python module can be found in test_MNIST_module1.py. This can be run using 

\# pytest test_MNIST_module1.py

## Image transformation module

Also included is a Python module for transformations of the images, which provide 
additional methods of data augmentation. This code can be found in imagetransformutils.py.

TODO: Add more functionality. Tests!

