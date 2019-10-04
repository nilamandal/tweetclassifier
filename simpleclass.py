from pandas import DataFrame, read_csv
import pandas as pd
import nltk

df = pd.read_csv('political.csv',header=0,usecols = ['message','message:confidence','text'],encoding = 'unicode_escape')
wanted= df.loc[(df['message'] == 'attack') | (df['message'] == 'support')]
print(wanted)
print(wanted['message'].value_counts())

df['bagofwords']=df['text']
