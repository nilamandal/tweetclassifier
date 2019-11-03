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

df = pd.read_csv('political.csv',header=0,usecols = ['message','message:confidence','text'],encoding = 'unicode_escape')
df= df.loc[(df['message'] == 'attack') | (df['message'] == 'support')]
df["textlength"]=df["text"].apply(len) #adding a column for length of post
df= df.loc[(df["textlength"]<280)] #removing posts that are too long

#removing bad characters and changing from string to list
df['text'] = df['text'].str.encode('ascii', 'ignore').str.decode('ascii')
df['sentiment']=df['text'].apply(getSentiment)
pd.set_option('display.max_colwidth', -1)
# df['text']=df['text'].apply(remove_punctuations)
# df['text']=df['text'].apply(str.lower)
# df['text']=df['text'].apply(str.split)
# df['text']=df['text'].apply(remove_stopwords)

# plt.hist(df['sentiment'], bins=50)
# plt.gca().set(title='Frequency Histogram, all', ylabel='Frequency');
# path="C:\\Users\\nila9\\Desktop\\UConn\\sentiments_all.png"
# plt.savefig(path)
df= df.loc[(df['sentiment'] != 0.0)]
print("overall sentiment scores mean and std dev:")
print(df['sentiment'].mean())
print(df['sentiment'].std())
dfs=df.loc[(df['message'] == 'support')]
print("support sentiment scores mean and std dev:")
print(dfs['sentiment'].mean())
print(dfs['sentiment'].std())
# plt.hist(dfs['sentiment'], bins=50)
# plt.gca().set(title='Frequency Histogram, support', ylabel='Frequency');
# path="C:\\Users\\nila9\\Desktop\\UConn\\sentiments_support.png"
# plt.savefig(path)
dfa=df.loc[(df['message'] == 'attack')]
print("attack sentiment scores mean and std dev:")
print(dfa['sentiment'].mean())
print(dfa['sentiment'].std())
# plt.hist(dfa['sentiment'], bins=50)
# plt.gca().set(title='Frequency Histogram, attack', ylabel='Frequency');
# path="C:\\Users\\nila9\\Desktop\\UConn\\sentiments_attack.png"
# plt.savefig(path)

# biglist=[]
# for x in df['text']:
#     biglist= biglist+x
# #print(biglist)
# dist= nltk.FreqDist(biglist)
#
# filtered_word_freq = dict((word, freq) for word, freq in dist.items() if not word.isdigit())
# sorted_x = sorted(filtered_word_freq.items(), key=operator.itemgetter(1))
# print(len(sorted_x))
# print(len(df))

#generateWordCloud(df['text'])
# print(count)
# objects = ("@",'http')
# y_pos = np.arange(len(objects))

# plt.bar(y_pos, counts, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('occurances')
# plt.xlabel("Links")
# plt.title('Link types in posts')
#
# path="C:\\Users\\nila9\\Desktop\\UConn\\length_support.png"
# plt.savefig(path)
