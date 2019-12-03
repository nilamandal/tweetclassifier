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
from xgboost import 

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve 

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
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweet_df['message'], test_size=0.73, random_state=42)
    tf_idf_vectorizer.fit(X_train)
    vector_train_X = tf_idf_vectorizer.transform(X_train)
    vector_test_X = tf_idf_vectorizer.transform(X_test)
    xgboost=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
    xgboost.fit(vector_train_X, y_train)
    predictions = xgboost.predict(vector_test_X)
    plot_data(xgboost, vector_test_X, y_test, predictions)



def classify_SVM(tweets_df):
    # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word",  min_df=5, use_idf=True, smooth_idf=True, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweet_df['message'], test_size=0.73, random_state=42)
    tf_idf_vectorizer.fit(X_train)
    vector_train_X = tf_idf_vectorizer.transform(X_train)
    vector_test_X = tf_idf_vectorizer.transform(X_test)
    SVC_classifier = SVC(probability=True)
    SVC_classifier.fit(vector_train_X, y_train)
    predictions = SVC_classifier.predict(vector_test_X)
    plot_data(SVC_classifier, vector_test_X, y_test, predictions)


def classify_logistic_regression(tweets_df):
    # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word",  min_df=5, use_idf=True, smooth_idf=True, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweet_df['message'], test_size=0.73, random_state=42)
    tf_idf_vectorizer.fit(X_train)
    vector_train_X = tf_idf_vectorizer.transform(X_train)
    vector_test_X = tf_idf_vectorizer.transform(X_test)
    logreg = LogisticRegression()
    logreg.fit(vector_train_X, y_train)
    predictions = logreg.predict(vector_test_X)
    plot_data(logreg, vector_test_X, y_test, predictions)

def classify_naive_bayes(tweets_df):
     # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word",  min_df=5, use_idf=True, smooth_idf=True, ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = train_test_split(tweets_df['text'], tweet_df['message'], test_size=0.73, random_state=42)
    tf_idf_vectorizer.fit(X_train)
    vector_train_X = tf_idf_vectorizer.transform(X_train)
    vector_test_X = tf_idf_vectorizer.transform(X_test)
    
    naive_bayes = MultinomialNB()
    naive_bayes.fit(vector_train_X, y_train)
    predictions = naive_bayes.predict(vector_test_X)
    
    plot_data(naive_bayes, vector_test_X, y_test, predictions)
   


tweet_df = load_data()

#classify_naive_bayes(tweet_df)
#classify_logistic_regression(tweet_df)
#classify_SVM(tweet_df)
classify_Xgboost(tweet_df)


