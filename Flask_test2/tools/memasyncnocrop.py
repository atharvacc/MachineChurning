
import warnings
from collections.abc import Sequence, Iterable
import numbers
from numpy import sin, cos, tan
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
import math
from PIL import Image
from skimage.io import imread
from os import listdir
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import pool
import multiprocessing as mp
import math
import os
import time

class Cropper:
    pil_img = 0
    img_arr = 0

    def __init__(self, input_img, save_dir, window_size=512, step_size=256):
        # globalfile.partitionOn = False

        self.input_img = input_img
        self.save_dir = save_dir
        self.window_size = window_size
        self.step_size = step_size
        
    # @timeit
    def runner(self):
       
        m, n, _ = self.input_img.shape

        window_size = 512
        step_size = 256
        n_row = (n/self.window_size) * (self.window_size/self.step_size) - 1
     
        x = 0
        y = 0

        pil_img = Image.fromarray(self.input_img)
        max_x = m - self.window_size
        max_y = n - self.window_size
        count = 0

        # Max_x = 3760
        # Max_y = 2336
        iter_ct_x = math.ceil(max_x / 256)
        iter_ct_y = math.ceil(max_y / 256)

        y_iterables = [(i * 256, i * 256 + 256) for i in range(iter_ct_y)]
        x_iterables = [(i * 256, i * 256 + 256) for i in range(iter_ct_x)]
        iterables = [[i, j] for i in x_iterables for j in y_iterables]
        # iterables
        final_iter = list(enumerate(iterables))

        count = 0

        with mp.Pool(mp.cpu_count()-1 or 1) as pool:
            pool.starmap_async(mycrop, final_iter).get()


    
    # @profile
    def mycrop(self,count, dim):
        x_dim = dim[0]
        y_dim = dim[1]
    

        x_lft = x_dim[0]
        x_rgt = x_dim[1]
        y_lft = y_dim[0]
        y_rgt = y_dim[1]
        window_size = 512
     
        sample1 = self.input_img[x_lft:x_lft+self.window_size, y_lft:y_lft+self.window_size, :]
        crop_img = Image.fromarray(sample1)
        crop_img.save("{}/Stack_{}.png".format(self.save_dir, str(count).zfill(4)))


