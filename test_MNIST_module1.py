#!/usr/bin/env python

import mnistutils as mn
import numpy as np
import pytest
import random
import string

def test_concat_matrix_lists():
    # create a couple of lists
    s1=np.random.randint(2, 10)
    s2=s1-1
    l1=np.random.randint(0, 10, s1)
    l2=np.random.randint(0, 10, s2)
    clist=mn.concat_matrix_lists(l1, l2)
    assert len(clist) == s1+s2

# TODO: this throws the following error:
# TODO: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
# TODO: fix ...

#def test_get_mnist_images_digit():
#    s1=np.random.randint(1, 20)
#    l1 = np.random.randint(0, 10, s1)
 #   l=mn.get_mnist_images_digit(l1)
 #   len2=len(l1)
 #   assert len2 == len(l)

def test_get_zero_matrix1():
    s1 = np.random.randint(2, 10)
    zm = mn.get_zero_matrix(s1)
    assert np.shape(zm)[1] == s1

# test passing the height & width parameters
def test_get_zero_matrix1():
    s1 = np.random.randint(2, 10)
    s2 = np.random.randint(2, 10)
    zm=mn.get_zero_matrix(s1, s2)
    assert np.shape(zm)[1] == s1
    assert np.shape(zm)[0] == s2

def test_get_random_spaces():
    _min = np.random.randint(2,20)
    _max = _min + np.random.randint(2,20)
    _length = np.random.randint(2,20)
    l = mn.get_random_spaces(_min, _max, _length)
    assert len(l) == _length

# this test is expected to fail
def test_get_random_spaces1():
    # we pass _max < _min
    _min = np.random.randint(2,20)
    _max = _min-1
    _length = np.random.randint(2,20)
    with pytest.raises(ValueError):
        l = mn.get_random_spaces(_min, _max, _length)

def test_shrink_or_grow_image1():
    _min = np.random.randint(2, 20)
    b1, b2 = mn.shrink_or_grow_image(_min)
    if _min < mn.mnist_dim:
        assert b1 == True
        assert b2 == False
    elif min > mn.mnist_dim:
        assert b1 == False
        assert b2 == True
    else:
        assert b1 == False
        assert b2 == False

def test_retrieve_images():
    s1 = np.random.randint(1, 50)
    l1 = np.random.randint(0, 10, s1)
    images=mn.retrieve_images(l1)
    assert len(l1) == len(images)

def test_shrink_matrix():
    # get a random length for our list of digits
    s1 = np.random.randint(1, 50)
    # generate a list of random digits
    l1 = np.random.randint(0, 10, s1)
    # go fetch our images ...
    images = mn.retrieve_images(l1)
    # so our MNIST images are 28x28
    # we *could* randomnly select an image from the list
    # but that's probably overkill, skince they're already
    # the result of a random list
    m = images[0]
    cut = np.random.randint(1, mn.mnist_dim)
    m1 = mn.shrink_matrix(m, cut)
    assert np.shape(m1)[0] == np.shape(m)[0]
    assert np.shape(m1)[1] == mn.mnist_dim - cut


def test_grow_matrix():
    # get a random length for our list of digits
    s1 = np.random.randint(1, 50)
    # generate a list of random digits
    l1 = np.random.randint(0, 10, s1)
    # go fetch our images ...
    images = mn.retrieve_images(l1)
    # so our MNIST images are 28x28
    # we *could* randomnly select an image from the list
    # but that's probably overkill, skince they're already
    # the result of a random list
    m = images[0]
    grow = np.random.randint(1, mn.mnist_dim)
    m1 = mn.grow_matrix(m, grow)
    assert np.shape(m1)[0] == np.shape(m)[0]
    assert np.shape(m1)[1] == mn.mnist_dim + grow

def test_check_digits1():
    l1 = np.random.randint(0, 10, 20)
    assert mn.check_input_numbers(l1) == True
    l1 = np.random.randint(0, 100, 20)
    assert mn.check_input_numbers(l1) == False
    l1 = np.random.randint(-10, 10, 20)
    assert mn.check_input_numbers(l1) == False

def test_check_input_type1():
    l1 = np.random.randint(0, 10, 20)
    assert mn.check_input_type(l1) == True

def test_check_input_type2():
    l=list(''.join(random.choices(string.ascii_uppercase + string.digits, k=20)))
    assert mn.check_input_type(l) == False

def test_checking_spacing():
    _min = np.random.randint(0, 10)
    _max = _min + 1
    spacing=(_min, _max)
    assert mn.check_spacing_range(spacing) == True
    _max = _min - 1
    spacing = (_min, _max)
    assert mn.check_spacing_range(spacing) == False

def test_check_width():
    assert mn.check_width('WRONGTYPE') == False
    width = np.random.randint(-100, 0)
    assert mn.check_width(width) == False
    width = np.random.randint(1, 100)
    assert mn.check_width(width) == True

