from skimage.io import imread_collection, imread
from skimage.io.collection import ImageCollection
from os import listdir

def list_files1(directory, extension):
    """
    list_files1: Find files with a certain extension in the directory and return the names in a list
            Args:
                directory: Directory to be searched
                extension: the extension of the files
            Returns:
                List of files with the extension within the directory
    """
    return list(( (directory + f) for f in listdir(directory) if f.endswith('.' + extension)))

def imread_grayscale(img_path):
    """
    imread_grayscale: Read images as gray scale
               Args:
                    img_path: Get the path of the image
               Returns:
                    Matrix with the image as a gray-scale
                
    """
    return imread(img_path, as_gray=True)

def load_from_dir(directory, extension):
    """
    load_from_dir: Takes in the directory path and extension of images to be read, and reads them in.
            Args:
                (String) directroy: path to directory
                (String) extension: extension of the files to be read
            Returns:
                (Numpy Array)data: The normal data w the 3 channels
                (Numpy Array) data_gray: the data converted to gray scale format 
    """
    data = np.array(ImageCollection('{a}/*.{b}'.format(a = directory, b = extension), load_func=imread))
    data_gray = np.array(ImageCollection('{a}/*.{b}'.format(a = directory, b = extension), load_func = imread_grayscale))
    return (data,data_gray)