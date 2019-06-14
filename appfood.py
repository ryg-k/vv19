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
from PIL import Image
import tensorflow as tf
import pandas as pd
import shutil
import bisect as bs
import pickle



font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()

graph = tf.get_default_graph()

json_file = open("ave.json", 'r')
model_json=json_file.read()
json_file.close()
model = model_from_json(model_json)
with graph.as_default():
    model.load_weights("weightsrm.hdf5")

app = Flask(__name__)


if not os.path.exists("uploads/"):
    os.makedirs("uploads/")
if not os.path.exists("static/graph/"):
    os.makedirs("static/graph/")
if os.path.exists("static/img_make/"):
    shutil.rmtree("static/img_make/")
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

def save_img1(img_data1):
    #色反転
    if request.method == 'POST':
        #ここでは、保存するだけ!
        #1
        if 'img_data1' not in request.files:
            flash('No file part')
            return redirect(request.url)
        img_data1=request.files['img_data1']
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

def plot_bar(labels, values, values2, imgname):
    height1 = values  # 点数1  # 点数2
    left = np.arange(len(height1))  # numpyで横軸を設定
    width = 0.3
    plt.bar(left, height1, color='skyblue', width=width,label="現状", align='center')
    plt.bar(left+width, values2, color='r', width=width,label="調整後", align='center')
    plt.xticks(left + width/2, labels)
    #plt.xticks(left , labels)
    plt.xlabel("栄養")
    plt.ylabel("割合(%)")
    plt.savefig(imgname)


def pred_image(img1):
    img = cv2.imread(img1, flags=cv2.IMREAD_UNCHANGED)
    #img = Image.open(img1)
    img = img / 255.0
    img_size = 50
    img = transform.resize(img, (img_size, img_size, 3), mode='constant')
    image = np.expand_dims(img, axis=0)
    with graph.as_default():
        predictions = model.predict(image)
    #predictionが予測したもの。

    labels_data = pd.read_csv("label.csv",encoding='utf-8')
    labelslist=(labels_data["name"])
    top5=np.array(predictions[0]).argsort()[-5:][::-1]
    print(top5)
    pred_result=[]
    for t in top5:
        pred_result.append(labelslist[t])
        print(labelslist[t])
        print(predictions[0][t])
    #pred_resultは予測した結果5つの料理名
    return pred_result , max(predictions[0])


def extract_nutrient_data(meal_name):
    #meal_nameは食事の名前 (英語)
    nutrient_csv = "foodlist_to_nutrient.csv"
    nutrient_data = pd.read_csv(nutrient_csv,encoding='utf-8')
    listpd=nutrient_data[nutrient_data['name']==meal_name]
    #listpdはpd
    head=list(nutrient_data.head(0))
    #headはlist
    #listpd[head[i]].values[0]のようにして、中身を取り出すことができる。
    return listpd , head


def pfc_judge(x0,Xp,sex,age,phys_act_level):
    #Xp is the dataframe of all food options with their values of all nutrients
    #x0 is the dataframe of food eaten with their values of all nutrients
    #initialization
    list_pfc = ['タンパク質(g)','脂質(kcal)','炭水化物(g)','エネルギー(kcal)']
    Xp = np.array(Xp.loc[:,list_pfc]*np.array([4,1,4,1])) #converting Xp to Xp[pfc] with the value indicating kcal
    x0 = np.array(x0.loc[list_pfc]).reshape(1,4)*np.array([4,1,4,1])
    N = Xp.shape[0] #number of syushoku options

    #initialization
    X = x0 + Xp #total pfc matrix (shaped N*4)
    Y = X[:,:3]/X[:,3].reshape(N,1) #%energy matrix (shaped N*3)

    #judge all the food
    Y0 = (0.13<=Y[:,0]) & (Y[:,0]<=0.2) #boolean matrix indicating if %energy of protein falls into the designated range
    Y1 = (0.2<=Y[:,1]) & (Y[:,1]<=0.3) #for fat
    Y2 = (0.5<=Y[:,2]) & (Y[:,2]<=0.65) #for carbonate
    ikinokori = np.where(Y0 * Y1 * Y2)[0] #array indicating where pfc ratio is in the designated range
    return ikinokori

def close_judge(x0,Xp,ikinokori,sex,age,phys_level):
    list_nutrient = np.array(['エネルギー(kcal)','タンパク質(g)',
    'n-6系 多価不飽和(g)','n-3系 多価不飽和(g)',
    '食物繊維 総量(g)','ビタミンA(μg)','ビタミンD(μg)','ビタミンE(mg)','ビタミンK(μg)',
    'ビタミンB1(mg)','ビタミンB2(mg)','ナイアシン(mg)','ビタミンB6(mg)','ビタミンB12(μg)',
    '葉酸(μg)','パントテン酸(mg)','ビオチン(μg)','ビタミンC(mg)','ナトリウム(mg)',
    'カリウム(mg)','カルシウム(mg)','マグネシウム(mg)','リン(mg)','鉄(mg)','亜鉛(mg)',
    '銅(mg)','マンガン(mg)','ヨウ素(μg)','セレン(μg)','クロム(μg)','モリブデン(μg)'])
    dict = {'male':0,'female':3}
    en_index = dict[sex] + phys_level
    #with open('energy.pickle', mode='rb') as g:
    #    energy = [pickle.load(g).iloc[age,en_index]]
    energy = [pd.read_pickle('energy.pickle').iloc[age,en_index]]
    #print(energy)
    #with open('nutrient_' + sex + '.pickle', mode='rb') as f:
    #    std_vect = np.array(energy + list(pickle.load(f).iloc[age,1:]))
    std_vect = np.array(energy + list(pd.read_pickle('nutrient_' + sex + '.pickle').iloc[age,1:]))
    #print(list(pd.read_pickle('nutrient_' + sex + '.pickle').iloc[age,1:]))
    zero_index = np.where(std_vect == 0)[0]
    std_vect = np.delete(std_vect,zero_index)
    list_nutrient = np.delete(list_nutrient,zero_index)
    Xp_std = ((Xp+x0).loc[:,list_nutrient])/std_vect
    lack = np.array([1 for i in range(std_vect.shape[0])]) - Xp_std
    #print(Xp_std)
    lack = lack.where(lack>0,lack/2)
    argmin = np.square(lack).sum(axis=1).idxmin()
    return argmin



def main1(x0,age,sex,phys_act_level):
    #load nutient data of syushoku
    #age: index
    #with open('dinner1.pickle',mode='rb') as f:
    #    dinner_df = pickle.load(f)
    dinner_df = pd.read_pickle('dinner1.pickle')
    dinner_name = dinner_df['foodname']
    Xp = dinner_df.iloc[:,1:] #dinner dataframe without names of food
    ikinokori = pfc_judge(x0,Xp,sex,age,phys_act_level)
    rec_food = close_judge(x0,Xp,ikinokori,sex,age,phys_act_level)
    total_nutrient = Xp.loc[rec_food] + x0
    return dinner_name[rec_food],total_nutrient

iglists=[]

SAVE_DIR="./static/img_make"
ans=[]
labels=[]
values1=[]
values2=[]
values3=[]

sum_nu=[]

ansre3=[]

val_ideal=[]

reccook=""
http_reccook=""

anssum=[]

@app.route('/')
def index():
    iglis=iglists
    #栄養
    print(iglis)
    a=os.listdir(SAVE_DIR)[::-1]
    print(a)
    madeimages=sorted(a,reverse=True)
    print(madeimages)
    madeimages_name=[]
    for name in madeimages:
        madeimages_name.append(name.replace(".jpg",""))
    graph_url=""
    if os.path.exists("static/graph/bar.png"):
        graph_url="static/graph/bar.png"
    print(graph_url)
    print(ans)
    return render_template('index.html',title="FOOD",images=madeimages,imagesname=madeimages_name,n=len(madeimages),
    iglis=iglis,graph_url=graph_url,ans=ans,labels=labels,values1=values1,values2=values2,values3=values3,
    val_ideal=val_ideal , reccook=reccook , http_reccook=http_reccook,ansre3=ansre3, anssum=anssum)

"""
@app.route('/confirm',methods=['POST'])
def form(display=None):
    query=request.files['img_data1']
    outcome=long_load(query)
    return render_template('done.html',display=outcome)
"""
@app.route('/confirm', methods = ['POST', 'GET'])
def save_img():

    #ここでは、保存するだけ!
    #1
    if 'img_data1' not in request.files:
        flash('No file part')
        return redirect(request.url)
    img_data1=request.files['img_data1']

    if request.form["sex"]=="1":
        sex="male"
        en0=0
    if request.form["sex"]=="2":
        sex="female"
        en0=3
    age=int(request.form["age"])+6
    print(age)
    phys_level=int(request.form["phys_act"])

    #画像ファイルかどうか判定
    #１つめだけ判定することにした
    count=0
    if img_data1:
        filename = secure_filename(img_data1.filename)
        basedir = os.path.abspath(os.path.dirname(__file__))
        #一旦保存する（必要あるかわからないが）
        img_data1.save(os.path.join(basedir,app.config['UPLOAD_FOLDER'], filename))
        MAIN_FILENAME = './uploads/' + filename
        #print(img_file)
        #画像読み込み

        img=Image.open(MAIN_FILENAME)
        width,height=img.size
        now=datetime.datetime.now()
        fmt_name = "{0:%Y-%m-%d-%H:%M:%S}.jpg".format(now)
        #時刻で名前つける

        if os.path.exists(os.path.join('static', 'img_make',fmt_name)):
            os.remove(os.path.join('static', 'img_make',fmt_name))
        img.save(os.path.join('static', 'img_make',fmt_name))
        img_url = os.path.join(app.config['UPLOAD_FOLDER'], fmt_name)
        img_url22=os.path.join('static', 'img_make',fmt_name)
        print(img_url)
        print(img_url22)

        #ansは投稿された画像の予測結果
        global ans
        ans , predrate=pred_image(MAIN_FILENAME)
        print("kooooooooooo")
        print(ans)
        print(predrate)
        print("kooooooooooo")
        #次に予測した結果に基づいて、栄養を表示
        #まず、top1の結果の栄養を表示
        listpd1 , head = extract_nutrient_data(ans[0])
        listpd2 , head = extract_nutrient_data(ans[1])
        listpd3 , head = extract_nutrient_data(ans[2])
        #listpd[head[i]].values[0]

        #レコメンド
        foodlist_df = pd.read_csv('foodlist_to_nutrient2.csv',index_col=0)
        x0 = foodlist_df.loc[ans[0]]*2 #data
        #ここで一回にどれくらい食べたか謎なため、とりあえず2倍しておく
        sum_nu.append(x0)
        if len(sum_nu) > 3:
            sum_nu.pop(0)
        sumnunu=sum(sum_nu)


        #時間かかる
        #文字列
        global reccook
        global http_reccook
        reccook,total_nutrient=main1(sumnunu,age,sex,phys_level)
        #total_nutrientは推奨したものも食べた後の栄養
        http_reccook=("https://www.google.com/search?q="+ reccook).replace(" ","")
        print(http_reccook)
        print("pppppppppppp")
        print(total_nutrient)
        print(total_nutrient[0])
        print(total_nutrient[1])
        print(total_nutrient[6])
        print(total_nutrient[2]/9)
        prednut=[total_nutrient[0],total_nutrient[1],total_nutrient[6],total_nutrient[2]/9]

        nus = [head[28],head[36],head[64],head[69]]
        #エネルギー(kcal),タンパク質(g),炭水化物(g),脂質(g)

        en_index = en0 + phys_level
        energy = [pd.read_pickle('energy.pickle').iloc[age,en_index]]
        n=pd.read_pickle('nutrient_' + sex + '.pickle').iloc[age,1:]

        v0=100*(listpd1[head[28]].values[0])/energy[0]
        v1=100*(listpd1[head[36]].values[0])/n[0]
        v2=100*(listpd1[head[64]].values[0])/(energy[0]*0.5/4)
        v3=100*(listpd1[head[69]].values[0])/(energy[0]*0.2/9)
        val0=[v0,v1,v2,v3]
        global val_ideal
        val_ideal=[int(energy[0]),int(n[0]),int(energy[0]*0.575/4),int(energy[0]*0.25/9)]
        #plot_bar(nus, val0, os.path.join('static', 'graph',"bar.png"))
        #ここで栄養を計算してリストでもっておく
        #head[28]=エネルギー(kcal)
        #head[50]=ビタミンC(mg)
        #head[64]=炭水化物(g)
        #head[72]=鉄(mg)
        #head[69]=脂質(g)
        #head[36]=タンパク質(g)

        #グラフで可視化する部分。
        global labels
        global values1
        labels = [head[28],head[36],head[64],head[69]]
        #clam_chowderは中身ないのでエラーになってしまう。
        values1 = [listpd1[head[28]].values[0],listpd1[head[36]].values[0],listpd1[head[64]].values[0],listpd1[head[69]].values[0]]
        global values2
        values2 = [listpd2[head[28]].values[0],listpd2[head[36]].values[0],listpd2[head[64]].values[0],listpd2[head[69]].values[0]]
        global values3
        values3 = [listpd3[head[28]].values[0],listpd3[head[36]].values[0],listpd3[head[64]].values[0],listpd3[head[69]].values[0]]
        print(predrate)

        ansre3.append([ans[0],listpd1[head[28]].values[0],listpd1[head[36]].values[0],listpd1[head[64]].values[0],listpd1[head[69]].values[0]])
        if len(ansre3) > 3:
            ansre3.pop(0)

        global anssum
        anssum=[0,0,0,0]
        for an in ansre3:
            if an[1] != an[1]:
                print("Yes")
                an[1]=0
                an[2]=0
                an[3]=0
                an[4]=0
            anssum[0]+=an[1]
            anssum[1]+=an[2]
            anssum[2]+=an[3]
            anssum[3]+=an[4]
        rate=[]
        print("kkkkkkkkkkkkkkkkkk")
        print(ansre3)
        print(anssum)
        print(val_ideal)
        for i in range(len(anssum)):
            rate.append(int(100*anssum[i]/val_ideal[i]))
        print(rate)
        print("kkkkkkkkkkkkkkkkkk2")
        print(prednut)
        prednurate=[]
        for i in range(len(prednut)):
            prednurate.append(int(100*prednut[i]/val_ideal[i]))
        print(prednurate)

        plot_bar(nus, rate, prednurate, os.path.join('static', 'graph',"bar.png"))
        iglists.append([fmt_name,ans[0],
        [labels[0],values1[0]],[labels[1],values1[1]],[labels[2],values1[2]],[labels[3],values1[3]],predrate])
        #plot_polar(labels, values1, os.path.join('static', 'graph',"radar.png"))
        return redirect('/')




if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0',port=5005)
