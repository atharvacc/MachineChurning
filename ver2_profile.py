

# import warnings
# from collections.abc import Sequence, Iterable
# import numbers
# from numpy import sin, cos, tan
# from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
# import math
from PIL import Image
from skimage.io import imread
# from os import listdir
# import numpy as np
# from multiprocessing.dummy import Pool as ThreadPool
# from multiprocessing import pool
# import multiprocessing as mp
# import math
from torchvision.transforms.functional import crop

# import os
# import time


@profile
def sim2():
    img1 = imread('./imgs/Rad.jpg')  # numpy.ndarray
    sample1 = img1[:512, :512, :]
    pil_img1 = Image.fromarray(sample1)


if __name__ == '__main__':
    sim2()
