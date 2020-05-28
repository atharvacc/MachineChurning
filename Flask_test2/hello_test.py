from flask import render_template, Flask, request, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename
import os
import glob
# from google.cloud import storage
import time
from tools.preprocess import Preprocessor
# from tools.preprocess_test import Preprocessor
from tools.postprocessor import Postprocessor
from skimage.io import imread
from PIL import Image
# import globalfile
from tools.util import gen_merged_horiz, gen_merged_vert,timeit

from tools.ganilla.predictor import Predictor

PREFIX = "https://storage.cloud.google.com/muse_app_data/"
BUCKET_NAME = "muse_app_data"
SAVE_DIR = "./imgs/test/"
RESULT_DIR = "./results/"

app = Flask(__name__)
# print(os.path.dirname(app.instance_path)) 
# # /mnt/d/coding/class/ecs193/MachineChurning/Flask_trial
# print(app.root_path) 
# #/mnt/d/coding/class/ecs193/MachineChurning/Flask_trial
# exit(1)

@timeit
@app.route('/')
def homepage():
    # Return a Jinja2 HTML template and pass in image_entities as a parameter.
    return render_template('hello.html')

@timeit
@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':
        # print(request.files)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return ("no file")
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return "No file"
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(filename))
            # public_url_input = push_to_bucket(os.path.join("folder", filename), BUCKET_NAME)
            generate_data(os.path.join(filename))
            # print()
            # predict("./imgs/")
            # output_url = generate_final_output(filename, 19, 256)
            # image_entity = [public_url_input, output_url]
            print("\n\n\n\n\nSENDING IMAGE: %s\n\n\n\n\n" %  os.path.join(file.filename))
            # print('%d %s cost $%.2f' % (6, 'bananas', 1.74)) 
            # print()
            image_entity = [os.path.join(file.filename)]
            
            # _,_,image_entity=os.walk(SAVE_DIR)
            # return render_template('hello.html')
            
            return render_template('hello_done.html', image_entities=image_entity)


# def push_to_bucket(local_file_name, bucket_name):
#     """
#     upload and set acl to public
#     """
#     year, month = time.strftime("%Y,%m").split(",")
#     dest_name = "{}-{}/{}".format(year, month, local_file_name.split("/")[-1])
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(dest_name)
#     blob.upload_from_filename(local_file_name)
#     blob.make_public()
#     return blob.public_url


@timeit
def generate_data(input_img_path):
    """
    preprocess data and generate data on which to predict
    """
    
    create_dir(SAVE_DIR)
    create_dir(RESULT_DIR)
    img = imread(input_img_path)
    m, n, o = img.shape
    window_size = 512
    step_size = 256
    n_row = (n/window_size) * (window_size/step_size) - 1
    Preprocess = Preprocessor(
        img, SAVE_DIR, window_size=window_size, step_size=step_size)
    Preprocess.generate_overlapping_images()

@timeit
def predict(img_dir):
    """
    predict on the preprocessed_data
    """
    model = Predictor(img_dir)
    model.predict()

@timeit
def create_dir(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    else:
        files = glob.glob(path_name+'*')
        for f in files:
            os.remove(f)

@timeit
def generate_final_output(orig_filename, row_count, step_size):
    """
    stack the predicted images and generate the final output
    
        Args:
            image_dir: Directory containing all of the predicted images
            row_count: number of images in every row
            step_size: The step size that was used the generate the images
    """

    Postprocess = Postprocessor(RESULT_DIR, row_count, step_size)
    out = Postprocess.stitch_blend()
    filename = "Prediction_" + orig_filename + ".png"
    Image.fromarray(out).save(filename)
    return filename
    # pred_url = push_to_bucket(filename, BUCKET_NAME)

    # Clear everything
    # os.system("rm -rf *.png results imgs ")
    # return pred_url