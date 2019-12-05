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
from scipy import stats
from collections import Counter
#nltk.download('punkt')
#nltk.download('stopwords')
'exec(%matplotlib inline)'
pd.set_option('display.max_colwidth', 100)

def load_bias_data():
    statistics_df, normalized_users, normalized_urls, normalized_hashtags=load_data(['bias','message','text'])
    
    return statistics_df

def load_audience_data():
    statistics_df, normalized_users, normalized_urls, normalized_hashtags=load_data(['audience','message','text'])
    return statistics_df

def load_data(columns):
    #read data
    data = pd.read_csv('files/political_media.csv',header=0,usecols = columns,encoding = 'unicode_escape')
    print("Check before filter political data")
    print(data.shape)
    # Remove text that's more than 280 characters
    data = data[data.text.apply(lambda x: len(str(x))<280)]
    # Get only rows that has (political or attack)
    data = filter_support_vs_attack(data)
    
    
    type_list = []
    users_list = []
    number_of_mentions_list = []

    hashtags_list = []
    number_of_hashtags_list = []
    tweet_length_list = []
    urls_list = []
    number_of_urls_list = []
    for index, row in data.iterrows():
        #build mentions,hashtags,urls dataframe
        users, number_of_mentions= get_mentions(row["text"])
        hashtags, number_of_hashtags = get_hashtags(row["text"])
        urls, number_of_urls = get_urls(row["text"])
        type_list.append(row['message'])
        users_list.append(users)
        number_of_mentions_list.append(number_of_mentions)
        hashtags_list.append(hashtags)
        number_of_hashtags_list.append(number_of_hashtags)
        urls_list.append(urls)
        tweet_length_list.append(len(row["text"]))
        number_of_urls_list.append(number_of_urls)

    normalized_users = get_normalized_statistics(type_list, users_list)
    normalized_urls = get_normalized_statistics(type_list, urls_list)
    normalized_hashtags = get_normalized_statistics(type_list, hashtags_list)

    statistics_df = DataFrame({"type":type_list, 
                    "users":users_list,
                    "number_of_mentions":number_of_mentions_list,
                    "hashtags":hashtags_list,
                    "number_of_hashtags":number_of_hashtags_list,
                    "urls":urls_list,
                    "number_of_urls":number_of_urls_list,
                    "tweet_length":tweet_length_list})  
    for col in columns: 
      if col not in ['message','text']:                 
        statistics_df[col] = data[col].values

    return statistics_df, normalized_users, normalized_urls, normalized_hashtags

def get_normalized_statistics(type_list, object_list):  
    
    all_object_list = []
    all_type_list =[]
    for type_class, user in zip(type_list, object_list):
        list_users = user.split(',')
        for i in list_users:
          all_type_list.append(type_class)
          all_object_list.append(i)
    object_df = DataFrame({"type":all_type_list, 
                    "object":all_object_list}) 
    return object_df

   
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

def plot_metadata_statistics():
    #bias_df = load_bias_data()
    #_plot_metadata_bias_statistics(bias_df, 'bias')
    audience_df = load_audience_data()
    _plot_metadata_audience_statistics(audience_df, 'audience')

def _plot_metadata_bias_statistics(meta_df, column_name):
    filtered_meta_df = meta_df[meta_df.type.isin(['support'])]
    print((filtered_meta_df[column_name].value_counts()/filtered_meta_df[column_name].count())*100)
    
    filtered_meta_df = meta_df[meta_df.type.isin(['attack'])]
    print((filtered_meta_df[column_name].value_counts()/filtered_meta_df['bias'].count())*100)
    plt.title( column_name.capitalize()+' in Support/Attack tweets')
    ind = np.arange(2)   
    width = 0.35       
    neutral=(83.311258,6.802721)
    partisian=(16.688742,93.197279)
    p1 = plt.bar(ind,neutral, width)
    p2 = plt.bar(ind, partisian, width,bottom=neutral)

    plt.ylabel('Type')
    plt.title(column_name.capitalize()+' for Support/Attack tweets')
    plt.xticks(ind, ('Neutral', 'Partisan'))
    plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Support', 'Attack'))
    plt.show()

def _plot_metadata_audience_statistics(meta_df, column_name):
    filtered_meta_df = meta_df[meta_df.audience.isin(['national'])]
    print((filtered_meta_df['type'].value_counts()/filtered_meta_df['type'].count())*100)
    
    filtered_meta_df = meta_df[meta_df.audience.isin(['constituency'])]
    print((filtered_meta_df['type'].value_counts()/filtered_meta_df['type'].count())*100)
    plt.title(column_name.capitalize()+' in Support/Attack tweets')
    ind = np.arange(2)   
    width = 0.35       
    national=(82.894737,93.939394)
    constituency=(17.105263,6.060606)
    p1 = plt.bar(ind,national, width)
    p2 = plt.bar(ind, constituency, width,bottom=national)

    plt.ylabel('Type')
    plt.title(column_name.capitalize()+' for Support/Attack tweets')
    plt.xticks(ind, ('National', 'Constituency'))
    plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Support', 'Attack'))
    plt.show()

def plot_normalized_statistics(object_df):
    object_df.groupby('type')['object'].nunique().plot(kind='bar')
    plt.show()

def plot_statistics(df):
    ax = plt.gca()
    df.groupby("type")['number_of_mentions'].mean().plot(kind='bar', color=["red","green"])
    plt.title('@ mentions in Support vs. Attack tweets')
    plt.xlabel('Type')
    plt.ylabel('Number of mentions') 
    plt.show()
    df.groupby("type")['number_of_hashtags'].mean().plot(kind='bar', color=["red","green"])
    plt.title('# Hashtags in Support vs. Attack tweets')
    plt.xlabel('Type')
    plt.ylabel('Number of hashtags')
    plt.show()
    df.groupby("type")['number_of_urls'].mean().plot(kind='bar', color=["red","green"])
    plt.title('URLS in Support vs. Attack tweets')
    plt.xlabel('Type')
    plt.ylabel('Number of urls')
    plt.show()
    


def plot_tweet_length(df):        
    df_tweet_length = df[['tweet_length','type']]
    df_support_tweet_length = df_tweet_length[df_tweet_length.type.isin(['support'])]
    df_attack_tweet_length = df_tweet_length[df_tweet_length.type.isin(['attack'])]
    
    #ax = df_support_tweet_length.plot.kde()
   
    fig, ax = plt.subplots()
    ax.hist(df_support_tweet_length["tweet_length"])
    plt.title('Tweet length in Support tweets')
    plt.xlabel('Tweet Length')
    plt.ylabel('Frequency')
    print("TWEET LENGTH SUPPORT")
    print(df_support_tweet_length.shape)
    plt.show()
    #ax = df_attack_tweet_length.plot.kde()
    #plt.title('Tweet length in Attack tweets')
    #plt.xlabel('Type')
    #plt.ylabel('Tweet Length')
    fig, ax = plt.subplots()
    ax.hist(df_attack_tweet_length["tweet_length"])
    plt.title('Tweet length in Attack tweets')
    plt.xlabel('Tweet Length')
    plt.ylabel('Frequency')
    print("TWEET LENGTH ATTACK")
    print(df_attack_tweet_length.shape)
    plt.show()

def get_most_common(df):
    attack_df = df[df.type.isin(['attack'])]
    support_df = df[df.type.isin(['support'])]
    print(pd.Series(' '.join(attack_df['object']).lower().split()).value_counts()[:10])
    print(pd.Series(' '.join(support_df['object']).lower().split()).value_counts()[:10])

#basic_columns = ['message','message:confidence','text']
#statistics_df, normalized_users, normalized_urls, normalized_hashtags = load_data(basic_columns)
#plot_tweet_length(statistics_df)
#plot_statistics(statistics_df)
#plot_normalized_statistics(normalized_users)
#plot_normalized_statistics(normalized_urls)
#plot_normalized_statistics(normalized_hashtags)

#get_most_common(normalized_users)
#get_most_common(normalized_urls)
#get_most_common(normalized_hashtags)
plot_metadata_statistics()




