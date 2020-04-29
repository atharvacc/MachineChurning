from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from torchvision.transforms.functional import crop
from PIL import Image
import seaborn as sns
import time
import globalfile


class Preprocessor:
    """
    Preprocessor implements generating overlapping image and saves it to a directory save_dir
    """
    # time , when reading certain size img that cost alot of time
    # switch to chunking
    # ie divide window in half

    def __init__(self, input_img, save_dir, window_size=512, step_size=256):
        globalfile.partitionOn = False

        self.input_img = input_img
        self.save_dir = save_dir
        self.window_size = window_size
        self.step_size = step_size

    # measures execution time

    def timeit(self, method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if 'log_time' in kw:
                name = kw.get('log_name', method.__name__.upper())
                kw['log_time'][name] = int((te - ts) * 1000)
            else:
                print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
            return result
            # time to run > 5 sec
            if kw['log_time'][name] > 5000:
                # chunk data
                globalfile.partitionOn = True
                globalfile.rep += 1
        return timed

    # 5120x5120 pixel size
    # 316 patches
    # 200-400 threads
    # 1.8Gb size

    # resulting patches stored in server buckets
    # stiched predictions stored in server buckets

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
        m, n, _ = self.input_img.shape
        x = 0
        y = 0
        pil_img = Image.fromarray(self.input_img)
        max_x = m - self.window_size
        max_y = n - self.window_size
        imgs = []
        count = 0
        while(x<=max_x):
            while(y<=max_y):
                crop_img = crop(pil_img, x, y, self.window_size, self.window_size)
                crop_img.save("{}/Stack_{}.png".format(self.save_dir, str(count).zfill(4)) )
                count=count+1
                y = y + self.step_size
            y = 0
            x = x + self.step_size
        return
