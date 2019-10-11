from pandas import DataFrame, read_csv
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.pyplot as plt; plt.rcdefaults()
from wordcloud import WordCloud

def remove_punctuations(text):
    if "http" in text:
        text= text.replace("http", "http ")
    punct= ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','[','\\',']','^','_','`','{','|','}','~']
    for p in punct:
        text = text.replace(p, '')
    return text

def remove_stopwords(textlist):
    stopwords=["a", "about", "above", "after", "afterwards", "again", "all", "almost", "alone", "along", "already", "also", "although", "am", "among",
    "amongst", "amoungst", "amount",  "an", "and", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as",  "at", "back", "be",
    "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between",
    "beyond", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do",
    "done", "down", "due", "during", "each", "eg", "eight", "either", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "fify", "fill", "find", "fire", "former", "formerly", "found", "from", "front", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "hence", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how",
    "however", "ie", "if", "in", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd",
    "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
    "name", "namely", "neither", "never", "nevertheless", "next", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off",
    "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per",
    "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "several", "should", "side", "since", "so", "some",
    "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take", "than", "that", "the", "their", "them", "themselves",
    "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "this", "those", "though",
    "three", "through", "throughout", "thru", "thus", "to", "together", "too", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
    "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whom", "whose", "why", "will", "with", "within", "without",
    "would", "yet", "you", "your", "yours", "yourself", "yourselves", 'i', 'im', 'amp']
    filtered_list = [value for value in textlist if value not in stopwords]
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

df = pd.read_csv('political.csv',header=0,usecols = ['message','message:confidence','text'],encoding = 'unicode_escape')
df= df.loc[(df['message'] == 'support') | (df['message'] == 'attack')]
df["textlength"]=df["text"].apply(len) #adding a column for length of post
df= df.loc[(df["textlength"]<300)] #removing posts that are too long

#removing bad characters and changing from string to list
df['text'] = df['text'].str.encode('ascii', 'ignore').str.decode('ascii')
#pd.set_option('display.max_colwidth', -1)
df['text']=df['text'].apply(remove_punctuations)
df['text']=df['text'].apply(str.lower)
df['text']=df['text'].apply(str.split)
df['text']=df['text'].apply(remove_stopwords)
#print(df['text'])

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
