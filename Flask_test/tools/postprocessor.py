from skimage.io import imread
import matplotlib.pyplot as plt 
import numpy as np
from os import listdir
from PIL import Image
import seaborn as sns
from .util import gen_merged_horiz, gen_merged_vert

class Postprocessor:
    """
    Postprocess the images by blending the overlapping the images and generating the original image again
    """
    def __init__(self, image_dir,row_count, step_size):
        self.image_dir = image_dir
        self.row_count = row_count
        self.step_size = step_size
    
    def stitch_blend(self):
        """
        stitch together the predictions made by the model
        
        Args:
            dir_name: directory containg all of the images
            param row_count: number of images in a row
        
        Returns:
                out: image where everything is stitched together
        """

        image_names = listdir(self.image_dir)
        image_names.sort()
        stackedImgs = []
        curStack = []
        cur_row = 1
        for img_name in image_names:
            img = imread(self.image_dir + img_name)
            curStack.append(img)
            if(cur_row % row_count == 0):
                hstacked = gen_merged_horiz(curStack, self.step_size)
                stackedImgs.append(hstacked)
                curStack=[]
                cur_row = 0
            cur_row= cur_row + 1
        out = gen_merged_vert(stackedImgs, self.step_size)
        return out

    
