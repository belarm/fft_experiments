#!/usr/bin/env python3
import matplotlib as mpl
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os.path
import argparse
import numpy as np
import cv2

class graph_display(tk.Frame):
    def __init__(self, parent, x):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self,text="FT display")
        label.pack(pady=10,padx=10)
        f = plt.figure(figsize=(10,5), dpi=100)

        canvas = FigureCanvasTkAgg(f, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(canvas, parent)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        a = Axes3D(f)
        x_width = x.shape[1]
        y_width = x.shape[0]
        Xs, Ys = np.meshgrid(np.arange(x_width), np.arange(y_width))
        a.plot_surface(Xs, Ys, x, alpha=0.7)

def close_window():
    quit() # Well, that works


parser = argparse.ArgumentParser(description='Display the generated fourier transform of (the grayscale of) an image')
parser.add_argument('image', help='The image file to transform')
parser.add_argument('-v', help='verbose mode', required=False, action='store_true')
args = parser.parse_args()
path = os.path.expanduser(args.image)


# The actual work. Really.
# Load image, calculate average value of collor channels
img = np.mean(cv2.imread(path),axis=2)
# Take n-dimensional real FT
ft = np.fft.rfftn(img)


# Draw it.
window = tk.Tk()
window.protocol('WM_DELETE_WINDOW', close_window)
gd = graph_display(window, np.real(ft))
gd = graph_display(window, np.imag(ft))
tk.mainloop()
