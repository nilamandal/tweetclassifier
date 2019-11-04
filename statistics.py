import re
import string
import nltk 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
from urlextract import URLExtract
nltk.download('punkt')
nltk.download('stopwords')
'exec(%matplotlib inline)'
pd.set_option('display.max_colwidth', 100)



def load_data():
    #read data
    data = pd.read_csv('files/political_media.csv',header=0,usecols = ['message','message:confidence','text'],encoding = 'unicode_escape')
    print("Check before filter political data")
    print(data.shape)
    # Remove text that's more than 300 characters
    data = data[data.text.apply(lambda x: len(str(x))<300)]
    # Get only rows that has (political or attack)
    data = filter_support_vs_attack(data)
    # get mentions of tweets
    
    type_list = []
    users_list = []
    number_of_mentions_list = []

    hashtags_list = []
    number_of_hashtags_list = []

    urls_list = []
    number_of_urls_list = []
    for index, row in data.iterrows():
        #build mentions dataframe
        users, number_of_mentions= get_mentions(row["text"])
        hashtags, number_of_hashtags = get_hashtags(row["text"])
        urls, number_of_urls = get_urls(row["text"])
        type_list.append(row['message'])
        users_list.append(users)
        number_of_mentions_list.append(number_of_mentions)
        hashtags_list.append(hashtags)
        number_of_hashtags_list.append(number_of_hashtags)
        urls_list.append(urls)
        number_of_urls_list.append(number_of_urls)


    statistics_df = DataFrame({"type":type_list, 
                    "users":users_list,
                    "number_of_mentions":number_of_mentions_list,
                    "hashtags":hashtags_list,
                    "number_of_hashtags":number_of_hashtags_list,
                    "urls":urls_list,
                    "number_of_urls":number_of_urls_list})  
         
    return statistics_df


def get_mentions(text):
    mentions = re.findall("(^|[^@\w])@(\w{1,15})", text)
    mentions = [user_name for rubbish, user_name in mentions]
    
    return ','.join(mentions) , len(mentions)

def get_hashtags(text):
    hashtags = [part[1:] for part in text.split() if part.startswith('#')]
    return ','.join(hashtags), len(hashtags)

def get_urls(text):
    extractor = URLExtract()
    urls = extractor.find_urls(text)
    return ','.join(urls), len(urls)

def filter_support_vs_attack(dataframe):
    filtered_dataframe = dataframe[dataframe.message.isin(['support', 'attack'])]
    print("Check filtered political data")
    print(filtered_dataframe.shape)
    return filtered_dataframe




statistics_df = load_data()

print(statistics_df.head())
