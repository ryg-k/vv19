import os

dir="food2/images/"
list0=[]
foodnames=[]
for folder in os.listdir(dir):
    foodnames.append(folder)
    fol=dir+folder
    #folder=apple_pie
    #fol=food-101/images/apple_pie
    count=0
    num= len([f for f in os.listdir(fol) if os.path.isfile(os.path.join(fol, f))])
    num1=num/3
    num2=5*num/9
    listtest=[]
    listtrain=[]
    listval=[]
    #fol=food-101/images/apple_pie
    for image in os.listdir(fol):
        #image=food-101/images/pizza/3261551.jpg
        if count<=num1:
            #img=dir+folder+'/'+image
            listtest.append(image)
            count=count+1
        if num1<count<=num2:
            #img=dir+folder+'/'+image
            listtrain.append(image)
            count=count+1
        if num2<count:
            #img=dir+folder+'/'+image
            listval.append(image)
            count=count+1
    list=[listtest,listtrain,listval]
    list0.append(list)
#foodnamesには、食べ物の名前が入っている(アルファベット順ではない)
#list0には、それぞれの食べ物について、test,traintrain,trainvalに
#分割するリスト。食べ物の順番は、foodnamesと同じ。
print(list0)
print(len(list0))
print(len(list0[0][0]))
print(len(list0[0][1]))
print(len(list0[0][2]))
#test = 1/3 traintrain = 2/9 trainval = 4/9

dir2="dataset/test/"
n=0
foodnamestest=[]
for folder in os.listdir(dir2):
    foodnamestest.append(folder)
    fol=dir2+folder
    for image in os.listdir(fol):
        if not image in list0[n][0]:
            img=dir2+folder+'/'+image
            os.remove(img)
    print(n)
    n=n+1
print(list0[100])
if foodnames == foodnamestest:
    print("yesssssssssssssssssssssssssssssssssss")

dir3="dataset/train/train/"
n=0
for folder in os.listdir(dir3):
    fol=dir3+folder
    for image in os.listdir(fol):
        if not image in list0[n][1]:
            img=dir3+folder+'/'+image
            os.remove(img)
    n=n+1
dir4="dataset/train/val/"
n=0
for folder in os.listdir(dir4):
    fol=dir4+folder
    for image in os.listdir(fol):
        if not image in list0[n][2]:
            img=dir4+folder+'/'+image
            os.remove(img)
    n=n+1
