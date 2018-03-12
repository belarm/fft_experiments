#!/usr/bin/env python3
import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def int_to_bits(x, lg_width=None):
    'Converts the supplied integer to a bitstring of length 2**_lg_width_, or the \
    smallest such bitstring which could represent it if _lg_width_ is not supplied'
    if lg_width is None:
        # HAXX
        lg_width = int(np.ceil(np.log2(np.ceil(np.log2(x)))))
    h = hadamard(2 ** lg_width)
    bitstring = np.full([2 ** lg_width], -1, dtype=np.int8)
    for i in range(2 ** lg_width): # (0,lg_width]
        if 2 ** i & x:
            bitstring[i] = 1
        else:
            bitstring[i] = -1
    return bitstring

def draw_func(x):
    plt.plot(x)
    # plt.show()

def draw_all_funcs(x):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_width = x.shape[1]
    y_width = x.shape[0]
    Xs, Ys = np.meshgrid(np.arange(x_width), np.arange(y_width))
    surf = ax.scatter3D(Xs, Ys, zs=x, alpha=1.)
    plt.show()


bits = np.array([int_to_bits(i,3) for i in range(256)])
# draw_all_funcs(bits)
h = hadamard(8)
# draw_all_funcs(bits @ h)
fts = bits @ h
# sample1 = fts[63]
# sample2 = fts[64]
# draw_func(fts[61])
# draw_func(fts[62])
# draw_func(fts[63])


# draw_func(fts[1])
# plt.show()

a = np.sum(np.outer(fts[2],fts[3]), axis=0)
print(fts @ h)

print(a)
