from flask import render_template, Flask, request, flash, redirect
from werkzeug.utils import secure_filename
import os
from google.cloud import storage
import time
from tools.preprocess import Preprocessor
# from tools.preprocess_test import Preprocessor
from tools.postprocessor import Postprocessor
from skimage.io import imread
from PIL import Image
from flask import render_template, Flask, request, flash, redirect, send_from_directory, url_for, jsonify
from multiprocessing import Process

from skimage.io import imread
import PIL
import glob
from shutil import copyfile
from tools.ganilla.predictor import Predictor
from markupsafe import escape

PREFIX = "https://storage.cloud.google.com/muse_app_data/"
BUCKET_NAME = "muse_app_data"
SAVE_DIR = "./imgs/test/"
RESULT_DIR = "./results/"

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config["APPLICATION_ROOT"] = os.path.abspath("")
print("app.config[\"APPLICATION_ROOT\"]: "+app.config["APPLICATION_ROOT"])
UPLOAD_FOLDER = os.path.join(app.config["APPLICATION_ROOT"], 'imgs')
print("UPLOAD_FOLDER: "+UPLOAD_FOLDER)
IMG_FOLDER = os.path.join(app.config["APPLICATION_ROOT"], 'static')

app.config['RESULTS_FOLDER'] = RESULT_DIR
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = IMG_FOLDER
print("STATIC_FOLDER: "+app.config['STATIC_FOLDER'])
app.static_folder = app.config['STATIC_FOLDER']


@app.route("/", methods=["GET", "POST"])
def homepage():
    global url1,url2
    messages = []
    inp1 = "" 
    inp2 = ""
    public_url_input=""
    output_url=""
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

            public_url_input = push_to_bucket(
                os.path.join(['STATIC_FOLDER'], stored_file_name), BUCKET_NAME)
            generate_data(os.path.join(
                app.config['STATIC_FOLDER'], stored_file_name))
            predict("./imgs/")
            output_url = generate_final_output(stored_file_name, 19, 256)
            # image_entity = [public_url_input, output_url]
            # test()
            url1="www.google.com"
            url2="www.yahoo.com"
            # inp1 = "https://www.ihcworld.com/imagegallery/images/he-stain/normal_Cerebellum-ms-g.jpg"
            # inp2 = "https://www.ihcworld.com/imagegallery/images/he-stain/normal_Colon-ms-g.jpg"
            messages = [public_url_input, output_url]

            print(
                "_________________________\n\n\n\n MESSAGES SET IN POST__________________\n\n"+str(messages)+"\n\n")
            # return render_template("index_final.html", messages=messages)
            # return jsonify({"redirect": "/test"})

            # print("")
            # return redirect(url_for('.testFunc', messages=messages))

            # return render_template("hello_final.html", image_entities=messages)

            # return render_template('hello_final.html', image_entities=image_entity)

    # return render_template("upload_file.html")
    # content = {'thing':'some stuff',
    #          'other':'more stuff'}
    # content = [inp1,inp2]
    # inp1 = "https://www.ihcworld.com/imagegallery/images/he-stain/normal_Cerebellum-ms-g.jpg"
    # inp2 = "https://www.ihcworld.com/imagegallery/images/he-stain/normal_Colon-ms-g.jpg"
    if url1 is not "":
        messages = [url1, url2]
    else:
        messages=[]
    if (len(messages) > 0):
        print("_________________________\n\n\n\n MESSAGES SET BEFORE RENDER__________________\n\n"+str(messages)+"\n\n")
        # return redirect(url_for('testFunc', messages=messages))
    
        # return render_template("hello_final.html", image_entities=messages,messages=messages)
        return render_template("index_final.html", image_entities=messages,messages=messages)
    else:
        print("_________________________\n\n\n\n MESSAGES No SET BEFORE RENDER__________________\n\n"+str(messages)+"\n\n")
        return render_template("index_final.html", image_entities=[],messages=[])


@app.route('/ques/<string:idd>', methods=['GET', 'POST'])
def ques(idd):
    print(idd)

url1=""
url2=""

@app.route('/test', methods=['GET', 'POST'])
def testFunc():
    inp1 = "https://www.ihcworld.com/imagegallery/images/he-stain/normal_Cerebellum-ms-g.jpg"
    inp2 = "https://www.ihcworld.com/imagegallery/images/he-stain/normal_Colon-ms-g.jpg"
    # messages = [inp1, inp2]
    dv_arr=[]
    if request.method == "POST":

        if request.args['value']:
            dv1 = request.args['value']
        else:
            dv1 = "" # counterpart for url_for()
        if request.args['value']:
            dv2 = request.args['value']
        else:
            dv2 = "" # counterpart for url_for()
        # dv2 = request.args['value'] or "empty" # counterpart for url_for()
        dv_arr=[dv1,dv2]
        print("_________________________\n\n\n\n DV_ARR RECEIVED __________________\n\n"+str(dv_arr)+"\n\n")
        # print(dv_arr)
    # image_entity = [messages[0], messages[1]]
    return render_template('hello_final.html', image_entities=dv_arr)


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
            predict("./imgs/")
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
    create_dir(RESULT_DIR)
    img = imread(input_img_path)
    m, n, o = img.shape
    window_size = 512
    step_size = 256
    n_row = (n/window_size) * (window_size/step_size) - 1
    Preprocess = Preprocessor(
        img, SAVE_DIR, window_size=window_size, step_size=step_size)
    Preprocess.generate_overlapping_images()


def predict(img_dir):
    """
    predict on the preprocessed_data
    """
    model = Predictor(img_dir)
    model.predict()


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

    Postprocess = Postprocessor(RESULT_DIR, row_count, step_size)
    out = Postprocess.stitch_blend()
    filename = "Prediction_" + orig_filename + ".png"

    Image.fromarray(out).save(filename)

    Image.fromarray(out).save(
        os.path.join(
            app.config['RESULTS_FOLDER'], filename))
    # file.save(os.path.join(
    #     app.config['STATIC_FOLDER'], stored_file_name))

    pred_url = push_to_bucket(filename, BUCKET_NAME)

    # Clear everything
    # os.system("rm -rf *.png results imgs ")
    return pred_url
