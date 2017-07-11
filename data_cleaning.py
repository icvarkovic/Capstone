import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, webtext
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import Counter

import sys

reload(sys)
sys.setdefaultencoding('utf8')

df= pd.read_csv('JEOPARDY_CSV.csv') #, encoding='latin')

to_drop= []
for index, question in enumerate(df[' Question']):
    if ' x ' in question or 'a href' in question:
        to_drop.append(index)

for index, value in enumerate(df[' Value']):
    if value not in set(['$100', '$200', '$300', '$400', '$500', '$600', '$800', '$1000', '$1200', '$1600', '$2000']):
        to_drop.append(index)
    elif df.loc[index, ' Air Date'] < '2001-11-26' and value not in set(['$100', '$200', '$300', '$400', '$500', '$600', '$800', '$1000']):
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
for index, value in enumerate(df[' Value']):
    if value not in set(['$200', '$400', '$600', '$800', '$1000', '$1200', '$1600', '$2000']):
        drop_more.append(index)
df.drop(df.index[drop_more], inplace= True)

def categorize(x):
    if x== '$200' or x== '$400':
        return 1
    elif x== '$600' or x=='$800':
        return 2
    elif x== '$1200' or x=='$1000':
        return 3
    else:
        return 4
df['Categories']= [categorize(x) for x in df['new_value']]

operators = set(('also', 'or', 'and', 'before', 'after'))
stop = set(stopwords.words('english')) - operators

list_of_docs = df['clean_question'].tolist()
list_of_answers= [answer.lower().split() for answer in df[' Answer']]
list_questions= [[word for word in doc.split() if word not in stop] for doc in list_of_docs]
j= zip(list_questions, list_of_answers)
questions= [x[0] + x[1] for x in j]         # joining answers and questions as one list
questions= [' '.join(q) for q in questions]
values= df['Categories'].tolist()

#calculating co-occurance
def dict_of_term_cooccurance(list_of_docs):
    term_freqs= {}
    for question in list_of_docs:
        for term in question.split():
            if term not in term_freqs:
                term_freqs[term]= Counter(question)
            else:
                term_freqs[term].update(question)
    return term_freqs

#for key in term_freqs:
    #term_freqs[key].pop(key)

#MODELING

def lemmatize_descriptions(questions):
    lem = WordNetLemmatizer()
    lemmatize = lambda d: " ".join([lem.lemmatize(word) for word in d.split()])
    return [lemmatize(desc) for desc in questions]


def get_vectorizer(questions):
    vect = TfidfVectorizer(max_features= 10000, ngram_range= (1,3))
    #vect = CountVectorizer(max_df= .6, min_df= 5)
    return vect.fit(questions)

q_train, q_test, y_train, y_test = train_test_split(questions, values)


vect = get_vectorizer(q_train)
X_train = vect.transform(lemmatize_descriptions(q_train)).toarray()
X_test = vect.transform(lemmatize_descriptions(q_test)).toarray()

#get top 5 features
def extract_top_features(feature_matrix):
    matrix= np.argsort(feature_matrix)[:,19995:]
    features= vect.get_feature_names()
    top_terms= []
    for line in matrix:
        top_terms.append([features[i] for i in line])
    return top_terms.toarray()

def calc_coocurrance(list_of_features):

'''Gets an array of top 5 most important features per question. Returns the sum of the
co-occurances for this group of features'''

    freqs= []
    for feature in list_of_features:
        freqs.append(np.array([term_freqs[feature][item] for item in list_of_features]))
    normalized= []
    for group in freqs:
        normed= np_group/np_group.max()
        normalized.append(sum(normed) - 1)
    return sum(normalized)

term_freqs= dict_of_term_cooccurance(q_train)
coocurrance_array= calc_coocurrance(extract_top_features(X_train))
X_train= np.append(X_train, coocurrance_array, axis= 1)
#term_freqs= dict_of_term_cooccurance(q_test)
#coocurrance_array= calc_coocurrance(top_terms.toarray())

X_train= np.append(X_train, coocurrance_array, axis= 1)
#X_train = pd.DataFrame(X_train, columns = vect.vocabulary_)
'''
model= MultinomialNB()
model.fit(X_train, y_train)
y_predict= model.predict(X_test)

print accuracy_score(y_test, y_predict)
