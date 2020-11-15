from flask import Flask, request, render_template, Response, redirect, url_for, g, current_app, send_file
import os
import json
from flask_bootstrap import Bootstrap
import requests
import pickle
import time
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from utils import face_rec, gen, encode_creation, functionist, gen2
from OpenSSL import SSL

basedir = os.path.abspath(os.path.dirname(__file__))

context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
context.use_privatekey_file('server.key')
context.use_certificate_file('server.crt')

app = Flask(__name__, )

bootstrap = Bootstrap(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir + "/data/", 'big.sql')
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
db = SQLAlchemy(app)
db.create_all()

list dir = ["data/models/dl/", "data/images/ANPR/data_gen/letters_from_videos/", "data/images/ANPR/data_gen/mask", "data/images/ANPR/data_gen/record_plates", "data/images/ANPR/data_gen/plate_loc/" , "data/images/ANPR/training", "data/images/ANPR/training/letters", "data/images/ANPR/training/mask_char_seg", "data/images/ANPR/training/mask_plate_loc", "data/images/ANPR/training/original_char_seg", "data/images/ANPR/training/orignal_plate_loc", "data/images/FaVe/enrolleds", "data/images/FaVe/snap/1", "data/notebooks", "data/vids"]
for MYDIR in list_dir:
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enrollment', methods=["GET", "POST"])
def page_red():
    t1 = time.time()

    if request.method == 'POST':
        params = request.get_json(force=False, silent=False, cache=True)
        
        encodes = pd.read_sql_table("Encodes", con=db.engine)
        id_num = len(encodes)
        encode_subj = functionist()
        encode_pd = encode_creation(encode_subj, id_num, params)
        row_to_write = list(encode_pd.loc[0])
        row_value_markers = ','.join(['?'] * len(row_to_write))
        db.engine.execute("INSERT INTO Encodes VALUES (%s)" % row_value_markers, row_to_write)

    t2 = time.time()

    return str(t2 - t1)


@app.route('/face_rec', methods=['POST'])
def face_recognition():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files.get('file')
            encodes = pd.read_sql_table("Encodes", con=db.engine)
            dict_res = face_rec(file, encodes)
            return json.dumps(dict_res)


@app.route('/id_cap', methods=["GET", "POST"])
def id_cap():
    return render_template("id_cap.html")


@app.route('/video_feed')
def video_feed():
    encodes = pd.read_sql_table("Encodes", con=db.engine)

    return Response(gen(encodes), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/anplr', methods=["GET", "POST"])
def anplr():
    return render_template("anplr.html")


@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_image', methods=['POST'])
def get_image():
    if request.method == 'POST':
        params = request.get_json(force=False, silent=False, cache=True)
        folio = params["folio"]
        name = "images/enrolleds/" + folio + "_5.png"
        return send_file(name)


@app.route('/test')
def test_stream():
    return render_template("test.html")


@app.errorhandler(404)
def page_not_found():
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
