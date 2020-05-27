
from flask import render_template, Flask, request, make_response, jsonify
from multiprocessing import Process

app = Flask(__name__)

# Define some heavy function


def my_func():
    # time.sleep(10)
    res = make_response(jsonify({"message": "File uploaded"}), 200)
    return res

@app.route("/", methods=["GET", "POST"])
def upload_file():

    if request.method == "POST":

        file = request.files["file"]
        
        print("File uploaded")
        print(file)

        heavy_process = Process(  # Create a daemonic process with heavy "my_func"
            target=my_func,
            daemon=True
        )
        heavy_process.start()

        # return res

    return render_template("upload_video.html")
