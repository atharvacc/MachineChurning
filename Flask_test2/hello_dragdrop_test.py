from flask import render_template, Flask, request, flash, redirect, send_from_directory, url_for, jsonify
from werkzeug.utils import secure_filename
import os
# from google.cloud import storage
import time
from tools.preprocess import Preprocessor
from multiprocessing import Process
# from tools.preprocess_test import Preprocessor
# from tools.postprocessor import Postprocessor
from skimage.io import imread
from PIL import Image
import PIL
# import globalfile
import glob
from shutil import copyfile
from tools.ganilla.predictor import Predictor
from markupsafe import escape
# from tools.ganilla.predictor import Predictor

PREFIX = "https://storage.cloud.google.com/muse_app_data/"
BUCKET_NAME = "muse_app_data"
SAVE_DIR = "./imgs/test/"
RESULT_DIR = "./results/"

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config["APPLICATION_ROOT"] = os.path.abspath("")
print("app.config[\"APPLICATION_ROOT\"]: "+app.config["APPLICATION_ROOT"])
UPLOAD_FOLDER = os.path.join(app.config["APPLICATION_ROOT"], 'imgs')
print("UPLOAD_FOLDER: "+UPLOAD_FOLDER)
IMG_FOLDER = os.path.join(app.config["APPLICATION_ROOT"], 'static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = IMG_FOLDER
print("STATIC_FOLDER: "+app.config['STATIC_FOLDER'])
app.static_folder = app.config['STATIC_FOLDER']


@app.route("/", methods=["GET", "POST"])
def homepage():
    # Return a Jinja2 HTML template and pass in image_entities as a parameter.
    if request.method == "POST":

        if 'file' not in request.files:
            flash('No file part')
            return ("no file")
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return "No file"
        else:
            
            stored_file_name = secure_filename(file.filename)

            file.save(os.path.join(
                app.config['STATIC_FOLDER'], stored_file_name))
            print("SAVED img STATIC path: " +
                  os.path.join(app.config['STATIC_FOLDER'], stored_file_name))
            copyfile(os.path.join(app.config['STATIC_FOLDER'], stored_file_name), os.path.join(
                SAVE_DIR, stored_file_name))
            image = PIL.Image.open(os.path.join(
                app.config['STATIC_FOLDER'], stored_file_name))  # image to open
            width, height = image.size
            print("image.size: " + str(width)+"x"+str(height))

            print("SAVE_DIR: " + SAVE_DIR)
            imgpath = str(SAVE_DIR) + str(file.filename)
            print("imgpath: "+imgpath)
            print("generate_data START")
            generate_data(os.path.join(
                app.config['STATIC_FOLDER'], stored_file_name))
        # return res

    # return render_template("upload_file.html")
    return render_template("index_final.html")


@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':
        # return jsonify(dict(redirect='path'))

        # print(request.files)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return ("no file")
        file = request.files['file']
        print("request.files: ")
        print(request.files)
        print("\n\n\n")
        if file.filename == '':
            flash('No selected file')
            return "No file"
        else:

            stored_file_name = secure_filename(file.filename)

            file.save(os.path.join(
                app.config['STATIC_FOLDER'], stored_file_name))
            print("SAVED img STATIC path: " +
                  os.path.join(app.config['STATIC_FOLDER'], stored_file_name))
            copyfile(os.path.join(app.config['STATIC_FOLDER'], stored_file_name), os.path.join(
                SAVE_DIR, stored_file_name))
            image = PIL.Image.open(os.path.join(
                app.config['STATIC_FOLDER'], stored_file_name))  # image to open
            width, height = image.size
            print("image.size: " + str(width)+"x"+str(height))

            print("SAVE_DIR: " + SAVE_DIR)
            imgpath = str(SAVE_DIR) + str(file.filename)
            print("imgpath: "+imgpath)
            print("generate_data START")
            generate_data(os.path.join(
                app.config['STATIC_FOLDER'], stored_file_name))
            # cut_imgs = os.listdir(os.path.join(app.config['UPLOAD_FOLDER']))
            # cut_imgs = cut_imgs[:5]
            # print("cut_imgs[:5]= "+str(cut_imgs))
            redirect(url_for('upload_photo'))
            print("return render_template(\"uploaded.html\",filename=stored_file_name)")
            return render_template("uploaded.html", filename=stored_file_name)


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


# def predict(img_dir):
#     """
#     predict on the preprocessed_data
#     """
#     model = Predictor(img_dir)
#     model.predict()


def create_dir(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    else:
        files = glob.glob(path_name+'*')
        for f in files:
            os.remove(f)


# def generate_final_output(orig_filename, row_count, step_size):
#     """
#     stack the predicted images and generate the final output

#         Args:
#             image_dir: Directory containing all of the predicted images
#             row_count: number of images in every row
#             step_size: The step size that was used the generate the images
#     """

#     Postprocess = Postprocessor(RESULT_DIR, row_count, step_size)
#     out = Postprocess.stitch_blend()
#     filename = "Prediction_" + orig_filename + ".png"
#     Image.fromarray(out).save(filename)
#     # pred_url = push_to_bucket(filename, BUCKET_NAME)

#     # Clear everything
#     os.system("rm -rf *.png results imgs ")
#     # return pred_url
