import pandas as pd
import numpy as np
import bisect as bs
import pickle

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
    with open('energy.pickle', mode='rb') as g:
        energy = [pickle.load(g).iloc[age,en_index]]
    with open('nutrient_' + sex + '.pickle', mode='rb') as f:
        std_vect = np.array(energy + list(pickle.load(f).iloc[age,1:]))
    zero_index = np.where(std_vect == 0)[0]
    std_vect = np.delete(std_vect,zero_index)
    list_nutrient = np.delete(list_nutrient,zero_index)
    Xp_std = ((Xp+x0).loc[:,list_nutrient])/std_vect
    lack = np.array([1 for i in range(std_vect.shape[0])]) - Xp_std
    lack = lack.where(lack>0,lack/2)
    argmin = np.square(lack).sum(axis=1).idxmin()
    return argmin

def main(x0,age,sex,phys_act_level):
    #load nutient data of syushoku
    #age: index
    with open('dinner1.pickle',mode='rb') as f:
        dinner_df = pickle.load(f)
    dinner_name = dinner_df['foodname']
    Xp = dinner_df.iloc[:,1:] #dinner dataframe without names of food
    #producing sample data (will be left out)
    a = np.random.randint(700000)
    ikinokori = pfc_judge(x0,Xp,sex,age,phys_act_level)
    return dinner_name[close_judge(x0,Xp,ikinokori,sex,age,phys_act_level)]

foodlist_df = pd.read_csv('foodlist_to_nutrient2.csv',index_col=0)
x0 = foodlist_df.loc['pad_thai']*2 #sample data
print(main(x0,age=8,sex='male',phys_act_level = 2))
#ageにはage groupのインデックスを入れる（何番目のグループか）
#sex は　'male' か 'female'のどちらか
#phys_act_level　は 1,2,3　のどれか
