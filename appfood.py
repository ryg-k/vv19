from flask import Flask,render_template, flash, request, redirect, url_for, send_from_directory
from PIL import Image, ImageDraw, ImageOps
import os
from werkzeug import secure_filename
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
import cv2
from keras.models import model_from_json, Sequential
from skimage import io, transform

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

app = Flask(__name__)

if not os.path.exists("uploads/"):
    os.makedirs("uploads/")
if not os.path.exists("static/graph/"):
    os.makedirs("static/graph/")
if not os.path.exists("static/img_make/"):
    os.makedirs("static/img_make/")
if os.path.exists("static/graph/radar.png"):
    os.remove("static/graph/radar.png")

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

def plot_polar(labels, values, imgname):
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
    values = np.concatenate((values, [values[0]]))  # 閉じた多角形にする
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-')  # 外枠
    ax.fill(angles, values, alpha=0.25)  # 塗りつぶし
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)  # 軸ラベル
    ax.set_rlim(0 ,250)
    fig.savefig(imgname)
    plt.close(fig)

def pred_image(img1):
    img = cv2.imread(img1, flags=cv2.IMREAD_UNCHANGED)
    json_file = open("ave.json", 'r')
    model_json=json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("weightsrm.hdf5")
    img = img / 255.0
    img_size = 50
    img = transform.resize(img, (img_size, img_size, 3), mode='constant')
    image = np.expand_dims(img, axis=0)
    predictions = model.predict(image)
    #predictionが予測したもの。

    labelslist=[]
    f=open("labels.txt")
    line=f.readline()
    labelslist.append(line.replace("\n",""))
    while line:
        line=f.readline()
        labelslist.append(line.replace("\n",""))
    top5=np.array(predictions[0]).argsort()[-5:][::-1]
    print(top5)
    for t in top5:
        print(labelslist[t])
        print(predictions[0][t])
    return labelslist[top5[0]]

iglists=[]
SAVE_DIR="./static/img_make"
@app.route('/')
def index():
    name="yokoyama"
    iglis=iglists
    a=os.listdir(SAVE_DIR)[::-1]
    print(a)
    a2=sorted(a,reverse=True)
    print(a2)
    graph_url=""
    if os.path.exists("static/graph/radar.png"):
        graph_url="static/graph/radar.png"
    print(graph_url)
    return render_template('index.html',title="FOOD",name=name,images=a2,iglis=iglis,graph_url=graph_url)

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
            now=datetime.datetime.now()
            fmt_name = "pic_{0:%Y%m%d-%H%M%S}.jpg".format(now)

            if os.path.exists(os.path.join('static', 'img_make',fmt_name)):
                os.remove(os.path.join('static', 'img_make',fmt_name))
            img2.save(os.path.join('static', 'img_make',fmt_name))
            img_url = os.path.join(app.config['UPLOAD_FOLDER'], fmt_name)
            img_url22=os.path.join('static', 'img_make',fmt_name)
            print(img_url)
            print(img_url22)
            #ここで栄養を計算してリストでもっておく
            iglists.append([fmt_name,"タンパク質","ビタミン","鉄分"])
            labels = ["タンパク質","ビタミン","鉄分","ビタミンB12"]
            values = [155, 156, 188, 139]
            plot_polar(labels, values, os.path.join('static', 'graph',"radar.png"))
            return redirect('/')




if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=5005)
