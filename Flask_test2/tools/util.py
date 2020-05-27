from skimage.io import imread
import matplotlib.pyplot as plt 
import numpy as np
from os import listdir
from PIL import Image
# import seaborn as sns
from functools import wraps
import time

def timeit(my_func):
    # f = open("myfile.txt", "a")
    
    @wraps(my_func)
    def timed(*args, **kw):

        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(
            my_func.__name__, (tend - tstart) * 1000))
        # f.write('"{}" took {:.3f} ms to execute\n'.format(
        #     my_func.__name__, (tend - tstart) * 1000))
            
        return output
    return timed

@timeit
def avg_blend(img1,img2):
    return ((img1+img2)/2).astype(np.uint8)

@timeit
def merge_horiz(left_img, right_img, step_size):
    """
    merge the two images using step size
    
    Args:
        left_img: the image on the left
        right_img: Image on the right
        Step_size: Step size used while generating the image
    Returns:
        combined_img: Image containing the combination of the two where the overlapping region
                      is merged using a blending function
    """
    m,n,_ = left_img.shape
    m1,n1,_ = right_img.shape
    left = left_img[:, 0:n-step_size,:]
    combine_left = left_img[:,n-step_size:,:].astype('float')
    combine_right = right_img[:, 0:n1-step_size,:].astype('float')
    right = right_img[:,n1-step_size:,:]
    combine = avg_blend(combine_left, combine_right)
    return  np.hstack((left, combine, right))

@timeit
def merge_vert(top_img, bot_img, step_size):
    """
    merge the two images using step size
    
    Args:
        left_img: the image on the left
        right_img: Image on the right
        Step_size: Step size used while generating the image
    Returns:
        combined_img: Image containing the combination of the two where the overlapping region
                      is merged using a blending function
    """
    m,n,_ = top_img.shape
    top = top_img[0:m-step_size,:,:]
    combine_top = top_img[m-step_size:,:,:].astype('float')
    combine_bot = bot_img[0:step_size,:,:].astype('float')
    bot = bot_img[step_size:,:,:]
    combine = avg_blend(combine_top, combine_bot)
    return  np.vstack((top, combine, bot))

@timeit
def gen_merged_horiz(img_list, step_size):
    """
    takes a list of images and merges them based on step_size
    
    Args:
        img_list: list containing all of the images
        step_size: the step size to be used 
    Returns:
        img: The merged image
    """
    
    img = img_list[0]
    new_list = img_list[1:]
    for img_r in new_list:
        img = merge_horiz(img, img_r, step_size)
    return img

@timeit
def gen_merged_vert(img_list, step_size):


    
    """
    takes a list of images and merges them based on step_size
    
    Args:
        img_list: list containing all of the images
        step_size: the step size to be used 
    Returns:
        img: The merged image
    """
    
    img = img_list[0]
    new_list = img_list[1:]
    for img_t in new_list:
        img = merge_vert(img, img_t, step_size)
    return img