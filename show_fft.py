#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path
import argparse
import numpy as np

def draw_all_funcs(x):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_width = x.shape[1]
    y_width = x.shape[0]
    Xs, Ys = np.meshgrid(np.arange(x_width), np.arange(y_width))
    surf = ax.plot_surface(Xs, Ys, x, alpha=1.)
    # plt.show()

parser = argparse.ArgumentParser(description='Display the generated fourier transform of an image')
parser.add_argument('ftfile', help='The transform to display')
parser.add_argument('-v', help='verbose mode', required=False, action='store_true')
args = parser.parse_args()
path = os.path.expanduser(args.ftfile)

ft = np.load(path)
draw_all_funcs(np.real(ft))
# plt.subplot(211)
draw_all_funcs(np.imag(ft))
plt.show()
