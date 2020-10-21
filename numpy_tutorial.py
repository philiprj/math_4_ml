
# coding: utf-8

import numpy as np # we commonly use the np abbreviation when referring to numpy

# New arrays can be made in several ways. We can take an existing list and convert it to a numpy array:

a = np.array([1,2,3])

# There are also functions for creating arrays with ones and zeros

np.zeros((2,2))

np.ones((3,2))


# ## Accessing Numpy Arrays
# You can use the common square bracket syntax for accessing elements
# of a numpy array

A = np.arange(9).reshape(3,3)
print(A)

print(A[0]) # Access the first row of A
print(A[0, 1]) # Access the second item of the first row
print(A[:, 1]) # Access the second column


# ## Operations on Numpy Arrays
# You can use the operations '*', '**', '\', '+' and '-' on numpy arrays and they operate elementwise.

a = np.array([[1,2], 
              [2,3]])
b = np.array([[4,5],
              [6,7]])

print(a + b)

print(a - b)

print(a * b)

print(a / b)

print(a**2)

print(a)
print(np.sum(a))


# Or sum along the first dimension
np.sum(a, axis=0)

# ## Linear Algebra
A = np.array([[2,4], 
             [6,8]])

# You can take transposes of matrices with `A.T`
print('A\n', A)
print('A.T\n', A.T)

# Note that taking the transpose of a 1D array has **NO** effect.

a = np.ones(3)
print(a)
print(a.shape)
print(a.T)
print(a.T.shape)

# But it does work if you have a 2D array of shape (3,1)

a = np.ones((3,1))
print(a)
print(a.shape)
print(a.T)
print(a.T.shape)

# ### Dot product

# We can compute the dot product between two vectors with np.dot
x = np.array([1,2,3])
y = np.array([4,5,6])
np.dot(x, y)


# We can compute the matrix-matrix product, matrix-vector product too.
# In Python 3, this is conveniently expressed with the @ syntax

A = np.eye(3) # You can create an identity matrix with np.eye
B = np.random.randn(3,3)
x = np.array([1,2,3])

# Matrix-Matrix product
A @ B

# Matrix-vector product
A @ x


# Sometimes, we might want to compute certain properties of the matrices.
# For example, we might be interested in a matrix's determinant, eigenvalues/eigenvectors.
# Numpy ships with the `numpy.linalg` package to do
# these things on 2D arrays (matrices).

from numpy import linalg

# This computes the determinant
linalg.det(A)

# This computes the eigenvalues and eigenvectors
eigenvalues, eigenvectors = linalg.eig(A)
print("The eigenvalues are\n", eigenvalues)
print("The eigenvectors are\n", eigenvectors)

# ### Time your code
# One tip that is really useful is to use the magic commannd `%time` to time the execution time of your function.


get_ipython().run_line_magic('time', 'np.abs(A)')

