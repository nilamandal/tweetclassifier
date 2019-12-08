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
from time import time
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

universal_test_size=0.7

def load_data():
    #read data
    data = pd.read_csv('political.csv',header=0,usecols = ['message','message:confidence','text'],encoding = 'unicode_escape')
    data["textlength"]=data["text"].apply(len) #adding a column for length of post
    data= data.loc[(data["textlength"]<280)] #removing posts that are too long
    data['text'] = data['text'].str.encode('ascii', 'ignore').str.decode('ascii')
    #print("Check before filter political data")
    #print(data.shape)
    # Get only rows that has (political or attack)
    data = filter_support_vs_attack(data)
    data['text']=data['text'].apply(preprocess_tweet)
    return data

def get_balanced_tweets(tweets_df):
    support_tweets = tweets_df[tweets_df.message.isin(['support'])]
    attack_tweets =  tweets_df[tweets_df.message.isin(['attack'])]

    training_support_tweets = support_tweets.iloc[300:400, :]
    training_attack_tweets = attack_tweets.iloc[0:100, :]

    testing_support_tweets = support_tweets.iloc[0:47, :]
    testing_attack_tweets = attack_tweets.iloc[100:147, :]

    balanced_training_tweets = training_attack_tweets.append(training_support_tweets)
    balanced_testing_tweets = testing_attack_tweets.append(testing_support_tweets)

     
    return balanced_training_tweets, balanced_testing_tweets

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
    #print("Check filtered political data")
    #print(filtered_dataframe.shape)
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

def plot_data(model, vector_test_X, y_test, predictions):
    print("Accuracy score:" , accuracy_score(y_test, predictions))
    print("Precision score:" , precision_score(y_test, predictions, pos_label='support'))
    print("Recall score:" , recall_score(y_test, predictions, pos_label='support'))
    print(classification_report(y_test, predictions))

    y_pred_prob = model.predict_proba(vector_test_X)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label='support')
    # create plot
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    _ = plt.xlabel('False Positive Rate')
    _ = plt.ylabel('True Positive Rate')
    _ = plt.title('ROC Curve')
    _ = plt.xlim([-0.02, 1])
    _ = plt.ylim([0, 1.02])
    _ = plt.legend(loc="lower right")
    plt.show()

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob, pos_label='support')
    # create plot
    plt.plot(precision, recall, label='Precision-recall curve')
    _ = plt.xlabel('Precision')
    _ = plt.ylabel('Recall')
    _ = plt.title('Precision-recall curve')
    _ = plt.legend(loc="lower left")
    plt.show()

def classify_Xgboost(tweets_df):
    # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word",  min_df=5, use_idf=True, smooth_idf=True, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweet_df['message'], test_size=universal_test_size, random_state=42)
    tf_idf_vectorizer.fit(X_train)
    print("method: xgboost")
    print("train set size:",len(y_train))
    print("test set size:",len(y_test))
    vector_train_X = tf_idf_vectorizer.transform(X_train)
    vector_test_X = tf_idf_vectorizer.transform(X_test)
    xgboost=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
    xgboost.fit(vector_train_X, y_train)
    predictions = xgboost.predict(vector_test_X)
    confusion_matrix(y_test,predictions)
    plot_data(xgboost, vector_test_X, y_test, predictions)



def classify_SVM(tweets_df):
    # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word",  min_df=5, use_idf=True, smooth_idf=True, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweet_df['message'], test_size=universal_test_size, random_state=42)
    tf_idf_vectorizer.fit(X_train)
    print("method: svm")
    print("train set size:",len(y_train))
    print("test set size:",len(y_test))
    vector_train_X = tf_idf_vectorizer.transform(X_train)
    vector_test_X = tf_idf_vectorizer.transform(X_test)
    SVC_classifier = SVC(probability=True)
    SVC_classifier.fit(vector_train_X, y_train)
    predictions = SVC_classifier.predict(vector_test_X)
    confusion_matrix(y_test,predictions)
    plot_data(SVC_classifier, vector_test_X, y_test, predictions)


def classify_logistic_regression(tweets_df):
    # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word",  min_df=5, use_idf=True, smooth_idf=True, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweet_df['message'], test_size=universal_test_size, random_state=42)
    tf_idf_vectorizer.fit(X_train)
    print("method: logistic regression")
    print("train set size:",len(y_train))
    print("test set size:",len(y_test))
    vector_train_X = tf_idf_vectorizer.transform(X_train)
    vector_test_X = tf_idf_vectorizer.transform(X_test)
    logreg = LogisticRegression()
    logreg.fit(vector_train_X, y_train)
    predictions = logreg.predict(vector_test_X)
    confusion_matrix(y_test,predictions)
    plot_data(logreg, vector_test_X, y_test, predictions)

def classify_naive_bayes(tweets_df, train=None, test=None):
     # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word",  min_df=5, use_idf=True, smooth_idf=True, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweets_df['message'], test_size=0.25, random_state=42)
    if train is not None and test is not None:
        X_train= train['text']
        X_test = test['text']
        y_train = train['message']
        y_test = test['message']
    tf_idf_vectorizer.fit(X_train)
    vector_train_X = tf_idf_vectorizer.transform(X_train)
    vector_test_X = tf_idf_vectorizer.transform(X_test)
    
    naive_bayes = MultinomialNB()
    naive_bayes.fit(vector_train_X, y_train)
    predictions = naive_bayes.predict(vector_test_X)
    
    plot_data(naive_bayes, vector_test_X, y_test, predictions)

def classify_MLP(tweets_df):
    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word",  min_df=5, use_idf=True, smooth_idf=True, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweet_df['message'], test_size=universal_test_size, random_state=42)
    tf_idf_vectorizer.fit(X_train)
    vector_train_X = tf_idf_vectorizer.transform(X_train)
    vector_test_X = tf_idf_vectorizer.transform(X_test)
    print("method: MLP")
    print("train set size:",len(y_train))
    print("test set size:",len(y_test))
    mlp=MLPClassifier(alpha=1, max_iter=1000)
    mlp.fit(vector_train_X, y_train)
    predictions=mlp.predict(vector_test_X)
    confusion_matrix(y_test,predictions)
    plot_data(mlp, vector_test_X, y_test, predictions)

def confusion_matrix(true_vals, pred_vals):
    #truesupport, falsesupport, trueattack, falseattack
    #print(len(pred_vals))
    ts=0
    fs=0
    ta=0
    fa=0
    index=0
    s=0
    a=0
    for x in true_vals:
        y=pred_vals[index]
        if x=='support':
            s+=1
        if x=='attack':
            a+=1
        if x=='support' and y=='support':
            ts+=1
        elif x=='support' and y=='attack':
            fa+=1
        elif x=='attack' and y=='attack':
            ta+=1
        else:
            fs+=1
        index+=1
    print("True amount of support, attack:",s,a)
    print("true support:",ts)
    print("false support:",fs)
    print("true attack:",ta)
    print('false attack:',fa)

tweet_df = load_data()

balanced_training_df, balanced_testing_df = get_balanced_tweets(tweeter_df)
balanced_df = balanced_training_df.append(balanced_testing_df)
classify_naive_bayes(balanced_df, train=balanced_training_df, test=balanced_testing_df)
classify_logistic_regression(tweet_df)
classify_SVM(tweet_df)
classify_Xgboost(tweet_df)
classify_MLP(tweet_df)
