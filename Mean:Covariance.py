
# coding: utf-8

# PACKAGE: DO NOT EDIT THIS CELL
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('fivethirtyeight')
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import time
import timeit

get_ipython().run_line_magic('matplotlib', 'inline')
from ipywidgets import interact


# Next, we are going to retrieve Olivetti faces dataset.
# 
# When working with some datasets, before digging into further analysis, it is almost always
# useful to do a few things to understand your dataset. First of all, answer the following
# set of questions:
# 
# 1. What is the size of your dataset?
# 2. What is the dimensionality of your data?
# 
# The dataset we have are usually stored as 2D matrices, then it would be really important
# to know which dimension represents the dimension of the dataset, and which represents
# the data points in the dataset. 
# 
# __When you implement the functions for your assignment, make sure you read
# the docstring for what each dimension of your inputs represents the data points, and which 
# represents the dimensions of the dataset!__. For this assignment, our data is organized as
# __(D,N)__, where D is the dimensionality of the samples and N is the number of samples.

image_shape = (64, 64)
# Load faces data
dataset = fetch_olivetti_faces('./')
faces = dataset.data.T

print('Shape of the faces dataset: {}'.format(faces.shape))
print('{} data points'.format(faces.shape[1]))

def show_face(face):
    plt.figure()
    plt.imshow(face.reshape((64, 64)), cmap='gray')
    plt.show()

# ## 1. Mean and Covariance of a Dataset

# In this week, you will need to implement functions in the cell below which compute the mean and covariance of a dataset.
# 

def mean_naive(X):
    "Compute the mean for a dataset X nby iterating over the data points"
    # X is of size (D,N) where D is the dimensionality and N the number of data points
    D, N = X.shape
    mean = np.zeros((D,1))
    ### Edit the code; iterate over the dataset and compute the mean vector.
    for n in range(N):
        # Update the mean vector
        mean = mean + X[:,n].reshape(D,1)
    mean = mean / N
    ###
    return mean

def cov_naive(X):
    """Compute the covariance for a dataset of size (D,N) 
    where D is the dimension and N is the number of data points"""
    D, N = X.shape
    ### Edit the code below to compute the covariance matrix by iterating over the dataset.
    covariance = np.zeros((D, D))
    ### Update covariance
    covariance = ((X - mean_naive(X)) @ (X - mean_naive(X)).T) / N
    ###
    return covariance


def mean(X):
    "Compute the mean for a dataset of size (D,N) where D is the dimension and N is the number of data points"
    # given a dataset of size (D, N), the mean should be an array of size (D,1)
    # you can use np.mean, but pay close attention to the shape of the mean vector you are returning.
    D, N = X.shape
    ### Edit the code to compute a (D,1) array `mean` for the mean of dataset.
    mean = np.zeros((D,1))
    ### Update mean here
    mean = np.mean(X, axis=1, keepdims=True)
    ###
    return mean

def cov(X):
    "Compute the covariance for a dataset"
    # X is of size (D,N)
    # It is possible to vectorize our code for computing the covariance with matrix multiplications,
    # i.e., we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    # We challenge you to give a vectorized implementation without using np.cov, but if you choose to use np.cov,
    # be sure to pass in bias=True.
    D, N = X.shape
    ### Edit the code to compute the covariance matrix
    covariance_matrix = np.zeros((D, D))
    ### Update covariance_matrix here
    covariance_matrix = np.cov(X, rowvar=True, bias=True)

    ###
    return covariance_matrix

X_test = np.arange(6).reshape(2,3)
expected_test_mean = np.array([1., 4.]).reshape(-1, 1)
expected_test_cov = np.array([[2/3., 2/3.], [2/3.,2/3.]])
print('X:\n', X_test)
print('Expected mean:\n', expected_test_mean)
print('Expected covariance:\n', expected_test_cov)

np.testing.assert_almost_equal(mean(X_test), expected_test_mean)
np.testing.assert_almost_equal(mean_naive(X_test), expected_test_mean)

np.testing.assert_almost_equal(cov(X_test), expected_test_cov)
np.testing.assert_almost_equal(cov_naive(X_test), expected_test_cov)


# We now test that both implementation should give identical results running on the faces dataset.

np.testing.assert_almost_equal(mean(faces), mean_naive(faces), decimal=6)
np.testing.assert_almost_equal(cov(faces), cov_naive(faces))


# With the `mean` function implemented, let's take a look at the _mean_ face of our dataset!

def mean_face(faces):
    return faces.mean(axis=1).reshape((64, 64))

plt.imshow(mean_face(faces), cmap='gray');


# Loops in Python are slow, and most of the time you want to utilise the fast native code provided by Numpy without explicitly using
# for loops. To put things into perspective, we can benchmark the two different implementation with the `%time` function
# in the following way:


# We have some HUUUGE data matrix which we want to compute its mean
X = np.random.randn(20, 1000)
# Benchmarking time for computing mean
get_ipython().run_line_magic('time', 'mean_naive(X)')
get_ipython().run_line_magic('time', 'mean(X)')
pass

# Benchmarking time for computing covariance
get_ipython().run_line_magic('time', 'cov_naive(X)')
get_ipython().run_line_magic('time', 'cov(X)')
pass


# As you can see, using Numpy's functions makes the code much faster!
# Therefore, whenever you can use something that's implemented in Numpy, be sure that you take advantage of that.

# ## 2. Affine Transformation of Datasets
# In this week we are also going to verify a few properties about the mean and
# covariance of affine transformation of random variables.
#
# For this assignment, you will need to implement the `affine_mean` and `affine_covariance` in the cell below.

# GRADED FUNCTION: DO NOT EDIT THIS LINE
def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        x: ndarray, the mean vector
        A, b: affine transformation applied to x
    Returns:
        mean vector after affine transformation
    """
    ### Edit the code below to compute the mean vector after affine transformation
    affine_m = np.zeros(mean.shape) # affine_m has shape (D, 1)
    ### Update affine_m
    affine_m = A @ mean + b
    ###
    return affine_m

def affine_covariance(S, A, b):
    """Compute the covariance matrix after affine transformation
    Args:
        S: ndarray, the covariance matrix
        A, b: affine transformation applied to each element in X        
    Returns:
        covariance matrix after the transformation
    """
    ### EDIT the code below to compute the covariance matrix after affine transformation
    affine_cov = np.zeros(S.shape) # affine_cov has shape (D, D)
    ### Update affine_cov
    affine_cov = A@S@A.T
    ###
    return affine_cov



