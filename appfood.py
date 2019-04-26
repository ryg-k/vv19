from flask import Flask,render_template, flash, request, redirect, url_for, send_from_directory
from PIL import Image, ImageDraw, ImageOps
import os
from werkzeug import secure_filename
import datetime


app = Flask(__name__)

if not os.path.exists("uploads/"):
    os.makedirs("uploads/")
if not os.path.exists("static/images/"):
    os.makedirs("static/images/")
if not os.path.exists("static/img_make/"):
    os.makedirs("static/img_make/")

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
# 初期画面、img_makeに画像があれば表示
@app.route('/')
def index():
    name="yokoyama"
    iglis=iglists
    a=os.listdir(SAVE_DIR)[::-1]
    print(a)
    a2=sorted(a,reverse=True)
    print(a2)
    return render_template('index.html',title="FOOD",name=name,images=a2,iglis=iglis)

"""
# long_load
@app.route('/confirm',methods=['POST'])
def form(display=None):
    query=request.files['img_data1']
    outcome=long_load(query)
    return render_template('done.html',display=outcome)
"""
@app.route('/confirm', methods = ['POST', 'GET'])
def save_img():
    if request.method == 'POST':
        # ここでは、保存するだけ!
        #1
        if 'img_data1' not in request.files:
            flash('No file part')
            return redirect(request.url)
        img_data1=request.files['img_data1'] # 画像ファイルを取ってくる
        hate=request.form["hate"] # 嫌いなものを取り出す
        # 画像ファイルかどうか判定
        if img_data1 and allowed_file(img_data1.filename):
            print("kkkkkkkkkkkkkkkkkkkkkkkk")
            filename = secure_filename(img_data1.filename)
            basedir = os.path.abspath(os.path.dirname(__file__))
            # uploadに一旦保存する（必要あるかわからないが）
            img_data1.save(os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename))
            MAIN_FILENAME = './uploads/' + filename
            #print(img_file)
            # 保存した画像読み込み
            img=Image.open(MAIN_FILENAME)
            width,height=img.size

            # 色反転(テスト用)
            img2=ImageOps.invert(img)

            now=datetime.datetime.now()
            fmt_name = "pic_{0:%Y%m%d-%H%M%S}.jpg".format(now)

            # 編集した画像を保存
            # 作った画像はimg_makeに保存
            if os.path.exists(os.path.join('static', 'img_make',fmt_name)):
                os.remove(os.path.join('static', 'img_make',fmt_name))
            img2.save(os.path.join('static', 'img_make',fmt_name))
            img_url = os.path.join(app.config['UPLOAD_FOLDER'], fmt_name)
            img_url22=os.path.join('static', 'img_make',fmt_name)
            print(img_url)
            print(img_url22)
            # ここで栄養を計算してリストでもっておく
            iglists.append([fmt_name,"タンパク質","ビタミン","鉄分"])
            # [filename,123,1,34]をグラフにする
            return redirect('/')  # /に戻る




if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=5005)
