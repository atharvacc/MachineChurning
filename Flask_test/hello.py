from flask import render_template, Flask, request, flash, redirect
from werkzeug.utils import secure_filename
import os
from google.cloud import storage
import time

app = Flask(__name__)

@app.route('/')
def homepage():
    # Return a Jinja2 HTML template and pass in image_entities as a parameter.
    return render_template('hello.html')

@app.route('/upload_photo', methods = ['GET', 'POST'])
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
            image_entity = ["../folder/" + filename]
            print("image entitiy is {}".format(image_entity))
            return render_template('hello.html', image_entities = image_entity)

def push_to_bucket(file):
    """
    Initializes bucket and pushes file to bucket

    Args:
        file: image to be predicted

    """
    storage_client = storage.Client()
    year, month = time.strftime("%Y,%m").split(",")
    bucket_name = "{}-{}".format(year,month)



