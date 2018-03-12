#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description='Generate the Fourier transform of an image and save it to a numpy array')
parser.add_argument('image', help='The image file to transform')
parser.add_argument('outfile', help='The output file to write to. Defaults to <image>.npy', nargs='?')
parser.add_argument('-v', help='verbose mode', required=False, action='store_true')
args = parser.parse_args()
path = os.path.expanduser(args.image)
if args.outfile is None:
    outfile = path + '.npy'

img = np.mean(cv2.imread(path),axis=2)
# print(img.shape)
# print(np.split(img,axis=2))
# out = np.zeros(img.shape, dtype=np.complex128)
# Calculate optimal FFT shape for FFTPACK
# fshape = [fftpack.helper.next_fast_len(int(d)) for d in img.shape]
# fslice = tuple([slice(0, int(sz)) for sz in shape])
# for i in range(img.shape[2]):
#     inp = img[:,:,i]
#     print(inp.shape)
#     res = np.fft.rfftn(inp)
#     print(res.shape)
    # out[:,:,i] =
# np.save(outfile, np.dstack([np.fft.rfftn(img[:,:,i]) for i in range(img.shape[2])]))
np.save(outfile, np.fft.rfftn(img))
if args.v:
    print('{} -> {}'.format(args.image, outfile))
# print(out.shape)
