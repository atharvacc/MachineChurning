from skimage.io import imread
from functools import wraps

# import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import os
import warnings
from collections.abc import Sequence, Iterable
import numbers
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import pool
import multiprocessing as mp
import math
# from torchvision.transforms.functional import crop
from PIL import Image
# import seaborn as sns
import time
import glob
# import globalfile
# import

try:
    import accimage
except ImportError:
    accimage = None


def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):

        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(
            my_func.__name__, (tend - tstart) * 1000))
        return output
    return timed


class Preprocessor:
    pil_img = 0
    img_arr = 0
    """
    Preprocessor implements generating overlapping image and saves it to a directory save_dir
    """
    @timeit
    def __init__(self, input_img, save_dir, window_size=512, step_size=256):
        # globalfile.partitionOn = False

        self.input_img = input_img  # is result of imread(input_img_path)
        self.save_dir = save_dir
        self.window_size = window_size
        self.step_size = step_size

    @timeit
    @classmethod
    def crop(cls, img, top, left, height, width):
        """Crop the given PIL Image.

        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            top (int): Vertical component of the top left corner of the crop box.
            left (int): Horizontal component of the top left corner of the crop box.
            height (int): Height of the crop box.
            width (int): Width of the crop box.

        Returns:
            PIL Image: Cropped image.
        """
        if not Preprocessor._is_pil_image(img):
            raise TypeError(
                "img should be PIL Image. Got {}".format(type(img)))

        return img.crop((left, top, left + width, top + height))

    @timeit
    @classmethod
    def _is_pil_image(cls, img):
        # global pil_img

        if accimage is not None:
            return isinstance(img, (Image.Image, accimage.Image))
        else:
            return isinstance(img, Image.Image)

    @timeit
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

        # with mp.Pool(mp.cpu_count()-1 or 1) as pool:
        with mp.Pool(2 or 1) as pool:
        # with mp.Pool(mp.cpu_count()-1 or 1) as pool:
            pool.starmap(self.mycrop, final_iter)
            # pool.starmap_async(self.mycrop, final_iter).get()

    # @profile
    @timeit
    def mycrop(self, count, dim):
        x_dim = dim[0]
        y_dim = dim[1]

        x_lft = x_dim[0]
        x_rgt = x_dim[1]
        y_lft = y_dim[0]
        y_rgt = y_dim[1]
        window_size = 512

        sample1 = self.input_img[x_lft:x_lft +
                                 self.window_size, y_lft:y_lft+self.window_size, :]
        crop_img = Image.fromarray(sample1)
        crop_img.save(
            "{}/Stack_{}.png".format(self.save_dir, str(count).zfill(4)))


    @timeit
    def generate_overlapping_images(self):
        """
        Generate crops from image of size window_size by window_size. Crops can be overlapping by
        setting the step_size

        Args:
            img: image as a numpy array
            window_size: Size of the crops that are generated
            step_size: step size for the window
        Returns:
            None
        """
        files = glob.glob(self.save_dir+'*')
        for f in files:
            os.remove(f)
        self.runner()
        # m, n, _ = self.input_img.shape
        # x = 0
        # y = 0
        # pil_img = Image.fromarray(self.input_img)
        # max_x = m - self.window_size
        # max_y = n - self.window_size
        # imgs = []
        # count = 0
        # while(x <= max_x):
        #     while(y <= max_y):
        #         crop_img = Preprocessor.crop(
        #             pil_img, x, y, self.window_size, self.window_size)
        #         crop_img.save(
        #             "{}/Stack_{}.png".format(self.save_dir, str(count).zfill(4)))
        #         count = count+1
        #         y = y + self.step_size
        #     y = 0
        #     x = x + self.step_size
        return

 # time , when reading certain size img that cost alot of time
    # switch to chunking
    # ie divide window in half

    # save_dir=$SAVE_DIR=
    #

        # 5120x5120 pixel size
        # 316 patches
        # 200-400 threads
        # 1.8Gb size

        # resulting patches stored in server buckets
        # stiched predictions stored in server buckets

    # measures execution time
       # def timeit(self, method):
        #     def timed(*args, **kw):
        #         ts = time.time()
        #         result = method(*args, **kw)
        #         te = time.time()
        #         if 'log_time' in kw:
        #             name = kw.get('log_name', method.__name__.upper())
        #             kw['log_time'][name] = int((te - ts) * 1000)
        #         else:
        #             print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        #         return result
        #         # time to run > 5 sec
        #         if kw['log_time'][name] > 5000:
        #             # chunk data
        #             globalfile.partitionOn = True
        #             globalfile.rep += 1
        #     return timed
