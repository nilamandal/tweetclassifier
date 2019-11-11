from pandas import DataFrame, read_csv
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.pyplot as plt; plt.rcdefaults()
from wordcloud import WordCloud
import operator
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def remove_punctuations(text):
    if "http" in text:
        text= text.replace("http", "http ")
    punct= ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','[','\\',']','^','_','`','{','|','}','~']
    for p in punct:
        text = text.replace(p, '')
    return text

def remove_stopwords(textlist):
    stop_words = list(set(stopwords.words('english')))
    filtered_list = [value for value in textlist if value not in stop_words]
    return filtered_list

def generateWordCloud(textframe):
    wcstarter=''
    for x in textframe:
        for y in x:
            wcstarter= wcstarter+' '+y
    wordcloud = WordCloud(background_color="white",max_font_size=40).generate(wcstarter)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    path="C:\\Users\\nila9\\Desktop\\UConn\\cloud_attack.png"
    plt.savefig(path)

def getSentiment(text):
    sid = SentimentIntensityAnalyzer()
    score=sid.polarity_scores(text)
    return score['compound']

df = pd.read_csv('political.csv',header=0,usecols = ['audience','bias','message','message:confidence','text'],encoding = 'unicode_escape')
df= df.loc[(df['message'] == 'attack') | (df['message'] == 'support')]
df["textlength"]=df["text"].apply(len) #adding a column for length of post
df= df.loc[(df["textlength"]<280)] #removing posts that are too long

#removing bad characters and changing from string to list
df['text'] = df['text'].str.encode('ascii', 'ignore').str.decode('ascii')
df['sentiment']=df['text'].apply(getSentiment)
pd.set_option('display.max_colwidth', -1)
df['text']=df['text'].apply(remove_punctuations)
df['text']=df['text'].apply(str.lower)
df['text']=df['text'].apply(str.split)
df['text']=df['text'].apply(remove_stopwords)

df_on= df.loc[(df["audience"]=='national')]
df_oc= df.loc[(df["audience"]=='constituency')]

#overall national vs constituency
plt.figure()
y_pos= np.arange(2)
performance=[len(df_on['audience']), len(df_oc['audience'])]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos,['national', 'constituency'])
plt.ylabel('number of tweets')
plt.title("overall: national vs constituency")
path="C:\\Users\\nila9\\Desktop\\UConn\\onc.png"
plt.savefig(path)
#attack national vs constituency
plt.figure()
df_an=df_on.loc[(df["message"]=='attack')]
df_ac=df_oc.loc[(df["message"]=='attack')]
performance=[len(df_an['audience']), len(df_ac['audience'])]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos,['national', 'constituency'])
plt.ylabel('number of tweets')
plt.title("attack: national vs constituency")
path="C:\\Users\\nila9\\Desktop\\UConn\\anc.png"
plt.savefig(path)

#support national vs constituency
plt.figure()
df_sn=df_on.loc[(df["message"]=='support')]
df_sc=df_oc.loc[(df["message"]=='support')]
performance=[len(df_sn['audience']), len(df_sc['audience'])]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos,['national', 'constituency'])
plt.ylabel('number of tweets')
plt.title("support: national vs constituency")
path="C:\\Users\\nila9\\Desktop\\UConn\\snc.png"
plt.savefig(path)


df_op= df.loc[(df["bias"]=='partisan')]
df_on= df.loc[(df["bias"]=='neutral')]
#overall partisan vs neutral
plt.figure()
y_pos= np.arange(2)
performance=[len(df_op['bias']), len(df_on['bias'])]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos,['partisan', 'neutral'])
plt.ylabel('number of tweets')
plt.title("overall: partisan vs neutral")
path="C:\\Users\\nila9\\Desktop\\UConn\\opn.png"
plt.savefig(path)
#attack partisan vs neutral
df_ap=df_op.loc[(df["message"]=='attack')]
df_an=df_on.loc[(df["message"]=='attack')]
plt.figure()
y_pos= np.arange(2)
performance=[len(df_ap['bias']), len(df_an['bias'])]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos,['partisan', 'neutral'])
plt.ylabel('number of tweets')
plt.title("attack: partisan vs neutral")
path="C:\\Users\\nila9\\Desktop\\UConn\\apn.png"
plt.savefig(path)
#support partisan vs neutral
df_sp=df_op.loc[(df["message"]=='support')]
df_sn=df_on.loc[(df["message"]=='support')]
plt.figure()
y_pos= np.arange(2)
performance=[len(df_sp['bias']), len(df_sn['bias'])]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos,['partisan', 'neutral'])
plt.ylabel('number of tweets')
plt.title("support: partisan vs neutral")
path="C:\\Users\\nila9\\Desktop\\UConn\\spn.png"
plt.savefig(path)
