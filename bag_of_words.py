import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
import operator
import string
import re
'exec(%matplotlib inline)'
pd.set_option('display.max_colwidth', 100)
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import islice

def load_data():
    #read data
    data = pd.read_csv('political.csv',header=0,usecols = ['message','message:confidence','text'],encoding = 'unicode_escape')
    preprocessed_tweets_text = []
    data["textlength"]=data["text"].apply(len) #adding a column for length of post
    data= data.loc[(data["textlength"]<280)] #removing posts that are too long
    data['text'] = data['text'].str.encode('ascii', 'ignore').str.decode('ascii')
    print("Check before filter political data")
    print(data.shape)
    # Get only rows that has (political or attack)
    data = filter_support_vs_attack(data)

    # preprocess text of tweets
    # for index, row in data.iterrows():
    #     row["text"] = preprocess_tweet(row["text"])
    #     print(row['text'])
    #     preprocessed_tweets_text.append(row["text"])
    data['text']=data['text'].apply(preprocess_tweet)
    preprocessed_tweets_text=data['text']
    #data['text']=data['text'].apply(str.split)
    #generate_bag_of_words(preprocessed_tweets_text)
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
    word_list1 = [word for word in nopunc if word not in stopwords.words('english')]
    special_words=['amp','house','rt','new','work','american']
    word_list = [word for word in word_list1 if word not in special_words]
    stemmer = SnowballStemmer('english')
    tokens_stemmed = [stemmer.stem(x) for x in word_list]
    #print(tokens_stemmed)
    return ' '.join(tokens_stemmed)

def filter_support_vs_attack(dataframe):
    filtered_dataframe = dataframe[dataframe.message.isin(['support','attack'])]
    print("Check filtered political data")
    print(filtered_dataframe.shape)
    return filtered_dataframe

def generate_bag_of_words(tweets_text):
    #print('All Tweets')
    #print(tweets_text)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tweets_text)
    #print('Bag of words Vectors for each Tweet')
    #print(len(X.toarray()))
    #print(X.toarray())
    print("Tweet words and its frequency in all tweets")
    print(vectorizer.vocabulary_ )
    filtered_word_freq = dict((word, freq) for word, freq in vectorizer.vocabulary_.items() if not word.isdigit())
    sorted_x = sorted(filtered_word_freq.items(), key=operator.itemgetter(1))
    print(sorted_x)

def generate_tfidf(tweet_text): #requires snowball to already be done
    cvec = CountVectorizer(stop_words='english', min_df=5, max_df=100, ngram_range=(1,2))
    cvec.fit(tweet_df.text)
    ngrams=list(islice(cvec.vocabulary_.items(), None))
    #ngramsdisplay=list(islice(cvec.vocabulary_.items(), 100))
    #print(len(ngrams))
    cvec_counts = cvec.transform(tweet_text)
    #print(cvec_counts)
    print( 'sparse matrix shape:', cvec_counts.shape)
    print( 'nonzero count:', cvec_counts.nnz)
    print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))
    occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
    counts_df = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
    #cc= counts_df.sort_values(by='occurrences', ascending=False).head(20)
    #print(cc)
    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(cvec_counts)
    #print(transformed_weights) #this is the actual vectors we want
    #print(transformed_weights.shape)
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    tfidf_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
    tfidf_df['occurences']= counts_df['occurrences']
    return tfidf_df

tweet_df = load_data()
pd.set_option('display.max_colwidth', -1)
#print(tweet_df['text'])
tfidf_df=generate_tfidf(tweet_df['text'])
cc= tfidf_df.sort_values(by='weight', ascending=False)
#print(cc)
#tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=100, stop_words=stopwords.words('english'))
#X = tfidfconverter.fit_transform(processed_tweets).toarray()
#tweet_df["tfidf"]=tweet_df["text"].apply(len)

# biglist=[]
# for x in tweet_df['text']:
#     biglist= biglist+x
#
# dist= nltk.FreqDist(biglist)
# filtered_word_freq = dict((word, freq) for word, freq in dist.items() if not word.isdigit())
# sorted_x = sorted(filtered_word_freq.items(), key=operator.itemgetter(1))
# print(sorted_x)
# print(len(sorted_x))
