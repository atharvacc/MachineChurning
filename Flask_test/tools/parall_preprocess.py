from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from torchvision.transforms.functional import crop
from PIL import Image
import seaborn as sns
from time import process_time
import multiprocessing as mp


# For 1 image time elapsed no parallelization= 89.351266

print("Number of processors: ", mp.cpu_count()) # 4 processors

class Preprocessor:
    """
    Preprocessor implements generating overlapping image and saves it to a directory save_dir
    """
    def __init__(self, input_img, save_dir, window_size = 512, step_size = 256):
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
        m,n,_ = self.input_img.shape
        x = 0
        y = 0
        pil_img = Image.fromarray(self.input_img)
        max_x = m - self.window_size
        max_y = n - self.window_size
        imgs = []
        count = 0
        x_coordinates = []
        y_coordinates = []
        #counter = 0
        while(x<=max_x):
            x_coordinates.append(x)
            x = x + self.step_size
            #counter = counter + 1

        #counter = 0
        while(y<=max_y):
            y_coordinates.append(y)
            y = y + self.step_size
            #counter = counter + 1

t1_start = process_time()
img = Image.open('/Users/moni/Downloads/full_input.png')
dir = "/Users/moni/Desktop/Output_imgs"

p = Preprocessor(np.array(img),dir)

p.generate_overlapping_images()

t1_stop = process_time()

print("Elapsed time: ", t1_stop-t1_start)
