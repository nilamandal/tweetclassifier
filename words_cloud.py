import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
nltk.download('punkt')
nltk.download('stopwords')

import string
import re
'exec(%matplotlib inline)'
pd.set_option('display.max_colwidth', 100)
import preprocessor as p
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import LancasterStemmer

from wordcloud import WordCloud

def load_data():
    #read data
    data = pd.read_csv('files/political_media.csv',header=0,usecols = ['message','message:confidence','text'],encoding = 'unicode_escape')
    preprocessed_tweets_text = []
    print("Check before filter political data")
    print(data.shape)
    # Remove text that's more than 280 characters
    data = data[data.text.apply(lambda x: len(str(x))<280)]
    # Get only rows that has (political or attack)
    data = filter_support_vs_attack(data)
    # preprocess text of tweets
    for index, row in data.iterrows():
        row["text"] = preprocess_tweet(row["text"])
        preprocessed_tweets_text.append(row["text"])

    return data

def preprocess_tweet(text):

    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # convert text to lower-case
    nopunc = nopunc.lower()
    # remove URLs
    nopunc = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', nopunc)
    nopunc = re.sub(r'http\S+', '', nopunc)
    # remove usernames
    nopunc = re.sub('@[^\s]+', '', nopunc)
    # remove the # in #hashtag
    nopunc = re.sub(r'#([^\s]+)', r'\1', nopunc)
    # remove repeated characters
    nopunc = word_tokenize(nopunc)
    # remove stopwords from final word list
    word_list = [word for word in nopunc if word not in stopwords.words('english')]
    #stem the words
    stemmer = SnowballStemmer('english')
    tokens_stemmed = [stemmer.stem(x) for x in word_list]
    sentence = ' '.join(tokens_stemmed) 
    return sentence

def filter_support_vs_attack(dataframe):
    filtered_dataframe = dataframe[dataframe.message.isin(['support', 'attack'])]
    print("Check filtered political data")
    print(filtered_dataframe.shape)
    return filtered_dataframe


def buildCloudText(tweet_df):
    attack_df = tweet_df[tweet_df.message.isin(['attack'])]
    support_df = tweet_df[tweet_df.message.isin(['support'])]

    attack_texts=[value['text'] for key,value in attack_df.iloc[:,2: ].iterrows()]
    support_texts=[value['text'] for key,value in support_df.iloc[:,2: ].iterrows()]

    generateWordCloud(attack_texts, "cloud_attack")
    generateWordCloud(support_texts, "cloud_support") 

def generateWordCloud(textframe, file_name):
    
    wcstarter=''
    for x in textframe:
        wcstarter= wcstarter+' '+x
    wordcloud = WordCloud(background_color="white",max_font_size=40).generate(wcstarter)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #path="C:\\Users\\nila9\\Desktop\\UConn\\cloud_attack.png"
    path = "Q:\\Azuz\\way to PHD\\Fall2019\\Social Media Analysis\\Project\\bag_of_words\\files\\"+file_name+".png"
    plt.savefig(path)

tweet_df = load_data()
buildCloudText(tweet_df)





