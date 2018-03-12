#!/usr/bin/env python3
import numpy as np
import cv2
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math import log, ceil
from scipy import fftpack, linalg
import threading

from numpy import (allclose, angle, arange, argsort, array, asarray,
                   atleast_1d, atleast_2d, cast, dot, exp, expand_dims,
                   iscomplexobj, mean, ndarray, newaxis, ones, pi,
                   poly, polyadd, polyder, polydiv, polymul, polysub, polyval,
                   product, r_, ravel, real_if_close, reshape,
                   roots, sort, take, transpose, unique, where, zeros,
                   zeros_like)

_rfft_mt_safe = True

_rfft_lock = threading.Lock()

def draw_all_funcs(x):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_width = x.shape[1]
    y_width = x.shape[0]
    Xs, Ys = np.meshgrid(np.arange(x_width), np.arange(y_width))
    surf = ax.plot_surface(Xs, Ys, x, alpha=1.)
    # plt.show()

def _inputs_swap_needed(mode, shape1, shape2):
    """
    If in 'valid' mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every dimension.
    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.
    Note that if the mode provided is not 'valid', False is immediately
    returned.
    """
    if mode == 'valid':
        ok1, ok2 = True, True

        for d1, d2 in zip(shape1, shape2):
            if not d1 >= d2:
                ok1 = False
            if not d2 >= d1:
                ok2 = False

        if not (ok1 or ok2):
            raise ValueError("For 'valid' mode, one must be at least "
                             "as large as the other in every dimension")

        return not ok1

    return False

def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = asarray(newshape)
    currshape = array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]


# Stolen from scipy
def fftconvolve(in1, in2, mode="full"):
    in1 = asarray(in1)
    in2 = asarray(in2)
    s1 = array(in1.shape)
    s2 = array(in2.shape)
    shape = s1 + s2 - 1

    # Check that input sizes are compatible with 'valid' mode
    if _inputs_swap_needed(mode, s1, s2):
        # Convolution is commutative; order doesn't have any effect on output
        in1, s1, in2, s2 = in2, s2, in1, s1

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [fftpack.helper.next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    sp1 = np.fft.rfftn(in1, fshape)
    sp2 = np.fft.rfftn(in2, fshape)
    draw_all_funcs(np.real(sp1))
    draw_all_funcs(np.real(sp2))
    # print(sp1)
    # print(sp2.shape)
    ret = (np.fft.irfftn(sp1 * sp2, fshape)[fslice].copy())
    draw_all_funcs(np.real(ret))
    draw_all_funcs(sp1 * sp2)
    plt.show()
    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

# First, a reference calculation from scipy
img = np.mean(cv2.imread('/home/belarm/source/nns/datasets/mnist/train/4/00002.png'),axis=2)
f = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
    ]) # Gaussian blur, un-normed
img2 = ndimage.convolve(img,f)
print(img)

#Now, given the same img and f, we should be able to:
# 1. Pad img by 1 pixel on all sides
# 2. Pad f to be the same size as the result
# 3. Take the FT of each.
# 4. Perform point-wise multiplication of the resultant matrices
# 5. Get img2 back from the inverse ft of the result.

#
# #TODO: THIS IS STILL WRONG
#
# size = 2 ** ceil(max(log(img.shape[0],2),log(img.shape[1],2)))
# # print(next_pow_of2)
# img_padded = np.pad(img, ((0,size-img.shape[0]), (0,size-img.shape[1])), mode='constant')
# filter_padded = np.pad(f, ((0,size-f.shape[0]),(0,size-f.shape[1])), mode='constant')
# print(img_padded.shape, filter_padded.shape)
# img_ft = np.fft.fftn(img_padded)
# filter_ft = np.fft.fftn(filter_padded)
# convolved = np.multiply(img_ft,filter_ft)
# img3 = np.imag(np.fft.ifft(convolved))
# print(img3.shape)
img3 = fftconvolve(np.pad(img, ((1,1),(1,1)),mode='symmetric'), f)[2:30,2:30]
# print(img2, img3)
# print(img3.shape)
# print(img2.shape)
# print(img2.astype(np.int)-img3.astype(np.int))
# plt.imshow(img2, cmap='gray')
# plt.show()
# plt.imshow(img3, cmap='gray')
# plt.show()
# plt.imshow(img2-img3, cmap='gray')
# plt.show()

# misc.imshow(img2)
