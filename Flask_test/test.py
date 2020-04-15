from flask import render_template, Flask, request, flash, redirect
from werkzeug.utils import secure_filename
import os
from google.cloud import storage
import time
from tools.preprocess import Preprocessor
from tools.postprocessor import Postprocessor
from skimage.io import imread
from PIL import Image

PREFIX = "https://storage.cloud.google.com/muse_app_data/"
BUCKET_NAME = "muse_app_data"
SAVE_DIR = "./imgs/test/"
RESULT_DIR = "./results/"

def create_dir(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)


def push_to_bucket(local_file_name, bucket_name):
    """
    upload and set acl to public
    """
    year, month = time.strftime("%Y,%m").split(",")
    dest_name = "{}-{}/{}".format(year,month, local_file_name.split("/")[-1])
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_name)
    blob.upload_from_filename(local_file_name)
    blob.make_public() 
    return blob.public_url

def generate_data(input_img_path):
    """
    preprocess data and generate data on which to predict 
    """
    create_dir(SAVE_DIR)
    create_dir(RESULT_DIR)
    img = imread(input_img_path)
    m,n,o = img.shape
    window_size = 512
    step_size = 256
    n_row = (n/window_size) * (window_size/step_size) - 1 
    Preprocess  = Preprocessor(img, SAVE_DIR, window_size = window_size, step_size = step_size)
    Preprocess.generate_overlapping_images()
    

def main():
    generate_data("./folder/full_input.png")

main()