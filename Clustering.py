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
from sklearn.cluster import KMeans
from time import time
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE

def load_data():
    #read data
    data = pd.read_csv('files/political_media.csv',header=0,usecols = ['message','message:confidence','text'],encoding = 'unicode_escape')
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
    return tfidf_df, transformed_weights

def generate_tf_idf(tweet_text):
     # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word",  min_df=5, use_idf=True, smooth_idf=True, ngram_range=(1, 2))
    tf_idf_matrix = tf_idf_vectorizer.fit_transform(tweet_text)

    return tf_idf_matrix

def cluster_tweets(tf_idf_matrix):
    num_clusters = 2
    num_seeds = 10
    # default value in SKILEARN
    max_iterations = 300 
    labels_color_map = {
        0: '#20b2aa', 1: '#ff7373'
    }
    pca_num_components = 2
    tsne_num_components = 2

    # create k-means model with custom config
    clustering_model = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations,
        precompute_distances="auto",
        n_jobs=-1
    )

    labels = clustering_model.fit_predict(tf_idf_matrix)
     
    X = tf_idf_matrix.todense()

    # ----------------------------------------------------------------------------------------------------------------------

    reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
    # print reduced_data

    fig, ax = plt.subplots()
    for index, instance in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
    
    ax.grid(True)
    plt.title('PCA Support and Attack Tweets')
    plt.show()



    # t-SNE plot
    embeddings = TSNE(n_components=tsne_num_components)
    Y = embeddings.fit_transform(X)
    plt.title('TSNE Support and Attack Tweets')
    plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
    ax.grid(True)
    plt.show()



tweet_df = load_data()
pd.set_option('display.max_colwidth', -1)
tf_idf_matrix = generate_tf_idf(tweet_df['text'])
cluster_tweets(tf_idf_matrix)

