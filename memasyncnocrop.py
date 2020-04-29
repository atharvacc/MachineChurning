
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

os.chdir("/mnt/d/coding/class/ecs193/MachineChurning")
save_dir = './imgs/test_parallel'

# os.getcwd()

print("Number of processors: ", mp.cpu_count())


# from torchvision.transforms.functional import crop
# import torch

# import seaborn as sns

# import torch

try:
    import accimage
except ImportError:
    accimage = None

PREFIX = "https://storage.cloud.google.com/muse_app_data/"
BUCKET_NAME = "muse_app_data"
RESULT_DIR = "./results/"

pil_img = 0
img_arr = 0

save_dir = './imgs/test_parallel'
# x_lft = 0 256 512 ..


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


# @profile
def mycrop(count, dim):
    x_dim = dim[0]
    y_dim = dim[1]
    global pil_img, img_arr
    #     curr_y=0
    #         count = 0
    # print(str(os.getpid())+str(x_dim) +str(x_dim[0])+","+str(x_dim[1])+str(y_dim)+str(y_dim[0])+","+str(y_dim[1]))
    #         print(x_dim[0])
    #         print(y_dim)

    x_lft = x_dim[0]
    x_rgt = x_dim[1]
    y_lft = y_dim[0]
    y_rgt = y_dim[1]
    window_size = 512
    #     img1 = imread('./MachineChurning/imgs/Rad.jpg')  # numpy.ndarray
    #   sample1 = img1[:512, :512, :]
    # pil_img1 = Image.fromarray(sample1)
    sample1 = img_arr[x_lft:x_lft+window_size, y_lft:y_lft+window_size, :]
    crop_img = Image.fromarray(sample1)
    # crop_img = crop(pil_img, x_lft, y_lft, window_size, window_size)
    crop_img.save("{}/Stack_{}.png".format(save_dir, str(count).zfill(4)))
    #         count=count+1
    #         curr_y +=256


def crop(img, top, left, height, width):
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
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    return img.crop((left, top, left + width, top + height))


def _is_pil_image(img):
    global pil_img, img_arr

    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


class Preprocessor:
    """
    Preprocessor implements generating overlapping image and saves it to a directory save_dir
    """
    global pil_img, img_arr

    def __init__(self, input_img, save_dir, window_size=512, step_size=256):
        self.input_img = input_img
        self.save_dir = save_dir
        self.window_size = window_size
        self.step_size = step_size

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
        m, n, _ = self.input_img.shape
        x = 0
        y = 0
        global pil_img, img_arr
        pil_img = Image.fromarray(self.input_img)
        max_x = m - self.window_size  #
        max_y = n - self.window_size
        imgs = []
        count = 0

        while x <= max_x:
            while y <= max_y:
                crop_img = crop(
                    pil_img, x, y, self.window_size, self.window_size)
                crop_img.save(
                    "{}/Stack_{}.png".format(self.save_dir,
                                             str(count).zfill(4))
                )
                count = count + 1
                y = y + self.step_size
            y = 0
            x = x + self.step_size
        return


@timeit
def runner():
    global pil_img, img_arr

    img = imread('imgs/Rad.jpg')  # numpy.ndarray
    img_arr = img

    m, n, o = img.shape
    window_size = 512
    step_size = 256
    n_row = (n/window_size) * (window_size/step_size) - 1
    #         Preprocess = Preprocessor(img, SAVE_DIR, window_size=window_size, step_size=step_size)
    #     Preprocess.generate_overlapping_images()

    x = 0
    y = 0

    Preprocess = Preprocessor(
        img, './imgs/test/', window_size=window_size, step_size=step_size)

    pil_img = Image.fromarray(Preprocess.input_img)
    max_x = m - Preprocess.window_size
    max_y = n - Preprocess.window_size
    imgs = []
    count = 0

    Max_x = 3760
    Max_y = 2336
    iter_ct_x = math.ceil(Max_x / 256)
    iter_ct_y = math.ceil(Max_y / 256)

    y_iterables = [(i * 256, i * 256 + 256) for i in range(iter_ct_y)]
    x_iterables = [(i * 256, i * 256 + 256) for i in range(iter_ct_x)]
    iterables = [[i, j] for i in x_iterables for j in y_iterables]
    # iterables
    final_iter = list(enumerate(iterables))

    count = 0

    with mp.Pool(10) as pool:
        #     print()
        # results = pool.starmap(mycrop, [(row, 4, 8) for row in data])
        pool.starmap_async(mycrop, final_iter).get()

        #     for i in final_iter:
        #         coord = i[1]
        #         print(coord)
        #         x_dim = coord[0]
        #         y_dim = coord[1]
        #         x_lft=x_dim[0]
        #         x_rgt=x_dim[1]
        #         y_lft=y_dim[0]
        #         y_rgt=y_dim[1]
        #         crop_img = crop(pil_img, x_lft, y_lft, window_size, window_size)
        #         crop_img.save("{}/Stack_{}.png".format(save_dir, str(count).zfill(4)))
        #         count +=1


def main():
    runner()


if __name__ == '__main__':
    main()
