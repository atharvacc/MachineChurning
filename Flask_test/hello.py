from flask import render_template, Flask, request, flash, redirect
from werkzeug.utils import secure_filename
import os
from google.cloud import storage
import time
from tools.preprocess import Preprocessor
from tools.postprocessor import Postprocessor
from skimage.io import imread
from PIL import Image
import globalfile

PREFIX = "https://storage.cloud.google.com/muse_app_data/"
BUCKET_NAME = "muse_app_data"
SAVE_DIR = "../ganilla/PREDICTION_DATA/testA"
temp_DIR = "../ganilla/PREDICTION_DATA/testB"
RESULT_DIR = "./results/"
MODEL_PATH = "../ganilla/saved_models/"
app = Flask(__name__)


@app.route('/')
def homepage():
    # Return a Jinja2 HTML template and pass in image_entities as a parameter.
    return render_template('hello.html')


@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':
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
            file.save(os.path.join("folder", filename))
            public_url_input = push_to_bucket(
                os.path.join("folder", filename), BUCKET_NAME)
            generate_data(os.path.join("folder", filename))
            predict()
            output_url = generate_final_output(filename, 19, 256)
            image_entity = [public_url_input, output_url]
            return render_template('hello.html', image_entities=image_entity)


def push_to_bucket(local_file_name, bucket_name):
    """
    upload and set acl to public
    """
    year, month = time.strftime("%Y,%m").split(",")
    dest_name = "{}-{}/{}".format(year, month, local_file_name.split("/")[-1])
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
    create_dir(temp_DIR)
    create_dir(RESULT_DIR)
    img = imread(input_img_path)
    m, n, o = img.shape
    window_size = 512
    step_size = 256
    n_row = (n/window_size) * (window_size/step_size) - 1
    Preprocess = Preprocessor(
        img, SAVE_DIR, window_size=window_size, step_size=step_size)
    try:
        Preprocess.generate_overlapping_images()
        # check runtimes
        if globalfile.partitionOn is True:

            if globalfile.rep > 1:
                window_size /= pow(2, globalfile.rep)
                step_size /= pow(2, globalfile.rep)

            window_size /= 2
            step_size /= 2
            if window_size <= 0 or step_size <= 0:
                raise Exception(
                    'window_size should not <= 0. The value of x was: {}'.format(window_size))
    except:
        pass

    os.system(
        "cp ../ganilla/PREDICTION_DATA/testA/Stack_0000.png ../ganilla/PREDICTION_DATA/testB/")


def predict():
    """
    predict on the preprocessed_data
    """
    os.system("python ../ganilla/test.py --epoch latest --results_dir {result_dir} --dataroot {data_path} \
                         --checkpoints_dir {model_path} --loadSize 512 --fineSize 512 --display_winsize 512 --name {test_model_name} \
                              --model cycle_gan --netG resnet_fpn".format(result_dir=RESULT_DIR, data_path="../ganilla/PREDICTION_DATA/", model_path=MODEL_PATH, test_model_name="test_cyclegan"))


def create_dir(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)


def generate_final_output(orig_filename, row_count, step_size):
    """
    stack the predicted images and generate the final output

        Args:
            image_dir: Directory containing all of the predicted images
            row_count: number of images in every row 
            step_size: The step size that was used the generate the images
    """

    os.system("cp -r ./results/test_cyclegan/test_latest/images ./results")
    # os.system("rm -rf results/test_cyclegan")
    create_dir("overlap")
    os.system("mv ./results/images/*_fake_B* ./overlap")

    Postprocess = Postprocessor("overlap/", row_count, step_size)
    out = Postprocess.stitch_blend()
    filename = "Prediction_" + orig_filename + ".png"
    Image.fromarray(out).save(filename)
    pred_url = push_to_bucket(filename, BUCKET_NAME)

    # Clear everything
    os.system("rm -rf *.png overlap results ../ganilla/PREDICTION_DATA")
    return pred_url
