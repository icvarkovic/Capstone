import string
import pandas as pd
import numpy as np
from unidecode import unidecode
import csv

import sys

reload(sys)
sys.setdefaultencoding('utf8')

df= pd.read_csv('JEOPARDY_CSV.csv') #, encoding='latin')
df= df[df[' Round']== 'Jeopardy!'].reset_index()

to_drop= []
for index, question in enumerate(df[' Question']):
    if ' x ' in question or 'a href' in question:
        to_drop.append(index)

for index, value in enumerate(df[' Value']):
    if value not in set(['$100', '$200', '$300', '$400', '$500', '$600', '$800', '$1000']):
        to_drop.append(index)
    elif df.loc[index, ' Air Date'] < '2001-11-26' and value not in set(['$100', '$200', '$300', '$400', '$500']):
        to_drop.append(index)
    else:
        continue

df.drop(df.index[list(set(to_drop))], inplace= True)

def transform_text(text):
    text = text.encode('ascii','ignore')
    #print type(text)
    new= text.translate(None, string.punctuation)
    return new.lower().strip()

def double(value):
    dollars= int(value[1:]) * 2
    value= '$' + str(dollars)
    return value

''' def unusual_words(text):
        text_vocab= set(w.lower() for w in text if w.isalpha())
        english_vocab= set(w.lower() for w in nltk.corpus.words.words())
        unusual= text_vocab.difference(english_vocab)
        return unusual'''


df['clean_question'] = df[' Question'].map(transform_text) #removes punctuation
mask= df[' Air Date'] < '2001-11-26'
new_df= df.loc[mask, ' Value'].map(double)
df.loc[df[' Air Date'] < '2001-11-26','new_value'] = new_df
df.loc[df[' Air Date'] >= '2001-11-26', 'new_value'] = df[' Value']  #equalizes $ values

drop_more= []
for index, value in enumerate(df['new_value']):
    if value not in set(['$200', '$400', '$600', '$800', '$1000']):
        drop_more.append(index)
df.drop(df.index[drop_more], inplace= True)

def categorize(x):
    if x== '$200':
        return 1
    elif x== '$400':
        return 2
    elif x== '$600':
        return 3
    elif x == '$800':
        return 4
    else:
        return 5

df['Categories']= [categorize(x) for x in df['new_value']]

df= df.reset_index()

df.to_csv(path_or_buf= 'clean_data_single_round.csv', index_label= False)
