import pandas as pd
import numpy as np
import bisect as bs
import json
from pprint import pprint

#load nutient data of syushoku
syushoku_df = pd.read_csv('dinner.csv').fillna(0)
syushoku_name = syushoku_df['foodname']
Xp = syushoku_df

#producing sample data
a = np.random.randint(400000)
x0 = Xp[a:a+1] #sample data
print(syushoku_name[a]) #name of the sample data

def pfc_judge(x0,Xp):
    #Xp is the dataframe of all food options with their values of all nutrients
    #x0 is the dataframe of food eaten with their values of all nutrients
    #initialization
    list_pfc = ['タンパク質(g)','脂質(kcal)','炭水化物(g)','エネルギー(kcal)']
    Xp = np.array(Xp.loc[:,list_pfc]*np.array([4,1,4,1])) #converting Xp to Xp[pfc] with the value indicating kcal
    x0 = np.array(x0.loc[:,list_pfc]).reshape(1,4)*np.array([4,1,4,1])

    #appling energy limit
    energy = 3000 - x0[0][3] #upper-bound for energy in a day
    Xp = Xp[np.where(Xp[:,3]<energy)] #eliminate food that exceeds the limit of energy
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

def EAR_UL_judge(sex,age,phys_act_level,x0):
    #list of nutrients used for judging lower-bound amounts
    list_kagen =  ['タンパク質(g)','ビタミンA(μg)','ビタミンB1(mg)','ビタミンB2(mg)','ナイアシン(mg)',
    'ビタミンB6(mg)','ビタミンB12(μg)','葉酸(μg)','ビタミンC(mg)','ナトリウム(mg)','カルシウム(mg)',
    'マグネシウム(mg)','鉄(mg)','亜鉛(mg)','銅(mg)','ヨウ素(μg)','セレン(μg)','モリブデン(μg)']

    #list of nutrients used for judging upper-bound amounts
    list_jogen = ['ビタミンA(μg)','ビタミンD(μg)','ビタミンE(mg)','ナイアシン(mg)','ビタミンB6(mg)',
    '葉酸(μg)','カルシウム(mg)','マグネシウム(mg)','リン(mg)','鉄(mg)','亜鉛(mg)','銅(mg)','マンガン(mg)',
    'ヨウ素(μg)','セレン(μg)','モリブデン(μg)']

    #list of all nutrients
    list_nutrient = ['エネルギー(kcal)','タンパク質(g)','脂質(kcal)',
    '脂肪酸 飽和(g)','n-6系 多価不飽和(g)','n-3系 多価不飽和(g)','炭水化物(g)',
    '食物繊維 総量(g)','ビタミンA(μg)','ビタミンD(μg)','ビタミンE(mg)','ビタミンK(μg)',
    'ビタミンB1(mg)','ビタミンB2(mg)','ナイアシン(mg)','ビタミンB6(mg)','ビタミンB12(μg)',
    '葉酸(μg)','パントテン酸(mg)','ビオチン(μg)','ビタミンC(mg)','ナトリウム(mg)',
    'カリウム(mg)','カルシウム(mg)','マグネシウム(mg)','リン(mg)','鉄(mg)','亜鉛(mg)',
    '銅(mg)','マンガン(mg)','ヨウ素(μg)','セレン(μg)','クロム(μg)','モリブデン(μg)']

    #load standard values
    EAR_df = pd.read_csv('EAR_male.csv').fillna(0)[age:age+1].loc[:,list_kagen] #EAR dataframe
    EAR_array = np.array(EAR_df)[0]
    UL_df = pd.read_csv('UL_male.csv').fillna(0)[age:age+1].loc[:,list_jogen]
    UL_array = np.array(UL_df)[0]

    #load sorted argument array
    arg_df = pd.read_csv('syushoku_argsort.csv')
    arg_EAR_df = arg_df.loc[:,list_kagen]
    arg_UL_df = arg_df.loc[:,list_jogen]
    arg_df = arg_df.loc[:,list_nutrient]
    arg_EAR_array = np.array(arg_EAR_df)
    arg_UL_array = np.array(arg_UL_df)
    arg_array = np.array(arg_df)

    #load syushoku data (and names)
    syushoku_df = pd.read_csv('syushoku.csv').fillna(0)
    syushoku_EAR_df = syushoku_df.loc[:,list_kagen]
    syushoku_UL_df = syushoku_df.loc[:,list_jogen]
    syushoku_name_array = syushoku_df['foodname']
    syushoku_df = syushoku_df.loc[:,list_nutrient]
    syushoku_array = np.array(syushoku_df)
    syushoku_EAR_array = np.array(syushoku_EAR_df)
    syushoku_UL_array = np.array(syushoku_UL_df)

    #select the food that satisfies the EAR
    food_EAR = np.array(x0.loc[:,list_kagen])[0]
    food_UL = np.array(x0.loc[:,list_jogen])[0]
    lack_vector = EAR_array - food_EAR #if lv[i]>0 then lack
    excess_vector = food_UL - UL_array #if ev[i]>0 then excess
    mask1 = np.array([])
    mask2 = np.array([])
    all = [i for i in range(len(arg_array[:,1]))]
    '''
    for i in range(len(lack_vector)): #EAR judge
        current_lack = lack_vector[i]
        arg_EAR_i = arg_EAR_array[:,i]
        syushoku_i = syushoku_EAR_array[:,i]
        syushoku_i = syushoku_i[arg_EAR_i]
        if current_lack>0:
            #print(list_kagen[i],lack_vector[i])
            a = bs.bisect_left(syushoku_i,current_lack)
            mask_i = arg_EAR_i[:a]
            #print(mask_i)
            mask1 = np.array(list((set(np.append(mask1,mask_i)))))
        ikinokori = (set(all)-set(mask1))
    print(ikinokori)
    '''

    for i in range(len(lack_vector)): #EAR judge
        current_ex = excess_vector[i]
        arg_UL_i = arg_UL_array[:,i]
        l = arg_UL_i.shape[0]
        syushoku_i = syushoku_UL_array[:,i]
        syushoku_i = syushoku_i[arg_UL_i]
        if current_ex>0:
            return np.array(set([]))
        else:
            a = bs.bisect_right(syushoku_i,-current_ex)
            mask_i = arg_UL_i[a:]
            mask2 = np.array(list((set(np.append(mask2,mask_i)))))
        ikinokori = set(all) - set(mask2)
    return np.array(list(ikinokori))

def close_judge(x0,Xp,ikinokori,sex,age,phys_level):
    list_nutrient = ['エネルギー(kcal)','タンパク質(g)','脂質(kcal)',
    '脂肪酸 飽和(g)','n-6系 多価不飽和(g)','n-3系 多価不飽和(g)','炭水化物(g)',
    '食物繊維 総量(g)','ビタミンA(μg)','ビタミンD(μg)','ビタミンE(mg)','ビタミンK(μg)',
    'ビタミンB1(mg)','ビタミンB2(mg)','ナイアシン(mg)','ビタミンB6(mg)','ビタミンB12(μg)',
    '葉酸(μg)','パントテン酸(mg)','ビオチン(μg)','ビタミンC(mg)','ナトリウム(mg)',
    'カリウム(mg)','カルシウム(mg)','マグネシウム(mg)','リン(mg)','鉄(mg)','亜鉛(mg)',
    '銅(mg)','マンガン(mg)','ヨウ素(μg)','セレン(μg)','クロム(μg)','モリブデン(μg)']
    with open('std_vects.txt') as f:
        std_vectors = json.load(f)
    std_vector = np.array(std_vectors[sex][age][phys_level])
    x0 = np.array(x0)[0]
    Xp = Xp + x0
    Xp = Xp.loc[:,list_nutrient]
    Xp_std = Xp/std_vector
    lack = np.array([1 for i in range(std_vector.shape[0])]) - Xp_std
    lack = lack.where(lack>0,0)
    argmin = lack.std(axis=1).idxmin()
    return argmin

ikinokori = pfc_judge(x0,Xp)
print(ikinokori)
print(syushoku_name[close_judge(x0,Xp,ikinokori,'male','18-30',1)])
#print(syushoku_name[pfc_judge(x0,Xp)])
#print(EAR_UL_judge(sex='male',age=10,phys_act_level=1,x0=x0))
#print(syushoku_name[EAR_UL_judge(sex='male',age=10,phys_act_level=1,x0=x0)])
