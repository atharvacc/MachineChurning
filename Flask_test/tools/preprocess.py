from skimage.io import imread
import matplotlib.pyplot as plt 
import numpy as np
from os import listdir
from torchvision.transforms.functional import crop
from PIL import Image
import seaborn as sns


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
        print("Max_x is {}".format(max_x))
        print("Max_y is {}".format(max_y))
        while(x<=max_x):
            while(y<=max_y):
                crop_img = crop(pil_img, x, y, self.window_size, self.window_size)
                crop_img.save("{}/Stack_{}.png".format(self.save_dir, str(count).zfill(4)) )
                count=count+1
                y = y + self.step_size
            y = 0 
            x = x + self.step_size
        return 
