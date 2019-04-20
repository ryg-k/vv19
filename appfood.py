from flask import Flask,render_template, flash, request, redirect, url_for, send_from_directory
import urllib
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import os
from werkzeug import secure_filename
import subprocess
import time

app = Flask(__name__)



UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['jpg','png','gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def long_load(typeback):
    time.sleep(5)
    return "You typed: %s" % typeback

iglists=[]
SAVE_DIR="./static/img_make"
@app.route('/')
def index():
    name="yokoyama"
    iglis=iglists
    return render_template('index.html',title="FOOD",name=name,images=os.listdir(SAVE_DIR)[::-1],iglis=iglis)

"""
@app.route('/confirm',methods=['POST'])
def form(display=None):
    query=request.files['img_data1']
    outcome=long_load(query)
    return render_template('done.html',display=outcome)
"""
@app.route('/confirm', methods = ['POST', 'GET'])
def save_img():
    if request.method == 'POST':
        #ここでは、保存するだけ!
        #1
        if 'img_data1' not in request.files:
            flash('No file part')
            return redirect(request.url)
        img_data1=request.files['img_data1']
        hate=request.form["hate"]
        #画像ファイルかどうか判定
        #１つめだけ判定することにした
        if img_data1 and allowed_file(img_data1.filename):
            print("kkkkkkkkkkkkkkkkkkkkkkkk")
            filename = secure_filename(img_data1.filename)
            basedir = os.path.abspath(os.path.dirname(__file__))
            #一旦保存する（必要あるかわからないが）
            img_data1.save(os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename))
            MAIN_FILENAME = './uploads/' + filename
            #print(img_file)
            #画像読み込み
            img=Image.open(MAIN_FILENAME)
            width,height=img.size
            img2=ImageOps.invert(img)
            if os.path.exists(os.path.join('static', 'img_make',filename)):
                os.remove(os.path.join('static', 'img_make',filename))
            img2.save(os.path.join('static', 'img_make',filename))
            img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_url22=os.path.join('static', 'img_make',filename)
            print(img_url)
            print(img_url22)
            #ここで栄養を計算してリストでもっておく
            iglists.append([filename,"タンパク質","ビタミン","鉄分"])
            return redirect('/')




if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=5005)
