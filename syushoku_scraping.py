import requests
from bs4 import BeautifulSoup
import pandas as pd
from pprint import pprint
j = 0
df = pd.DataFrame({'name':['first']})
for i in range(1,16):
    if i == 1:
        url = "http://calorie.slism.jp/category/16/"
    else:
        url = "http://calorie.slism.jp/category/16/" + str(i)
    r = requests.get(url)
    soup = BeautifulSoup(r.text,"lxml")
    soup_page = BeautifulSoup(r.text,"lxml")
    for a in soup.find_all("a",class_ = "soshoku_a"): #for all 主食
        j += 1
        food_name = str(a.string)
        r = requests.get("http://calorie.slism.jp/"+a['href'])
        soup_page = BeautifulSoup(r.text,"lxml")
        dict = {"name":[food_name]}
        i = 0
        for table in soup_page.find_all("table"):
            for tr in table.find_all("tr"):
                for td in tr.find_all("td","label"):
                    i+=1
                    if i <= 6:
                        continue
                    if td.next_sibling.string == '\n':
                        list = td.next_sibling.next_sibling.contents
                        string = (str(list[0].string)+str(list[1].string)) #str(list[1])
                    else:
                        string = str(td.next_sibling.string)
                    dict[str(td.string)] = [string]
                for td in tr.find_all("td","valMeasure"):
                    name = tr.find(class_ = "name")
                    if name != None:
                        if td.string == None:
                            string = ''
                            for children in td.children:
                                string += str(children.string)
                        else:
                            string = str(td.string)
                        dict[str(name.string)] = [string]
        df1 = pd.DataFrame(dict)
        df = df.append(df1)
        print(j)


df.replace('NaN',0)
df.to_csv('abc.csv',encoding='utf_8_sig')
    #print(pd.DataFrame(dict))
