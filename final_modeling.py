import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, webtext
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn import svm
import pandas as pd
import numpy as np
from collections import Counter
from unidecode import unidecode
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import sys

reload(sys)
sys.setdefaultencoding('utf8')

df = pd.read_csv('clean_data_single_round.csv')
df.drop(df.index[51187], inplace= True)

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
def dict_of_term_cooccurrence(list_of_docs):
    term_freqs= {}
    for question in list_of_docs:
        word_list= question.split()
        for term in word_list:
            if term in term_freqs:
                term_freqs[term].update(word_list)
            else:
                term_freqs[term]= Counter(word_list)
    return term_freqs

#for key in term_freqs:
    #term_freqs[key].pop(key)

#MODELING

def lemmatize_descriptions(questions):
    lem = WordNetLemmatizer()
    lemmatize = lambda d: " ".join([lem.lemmatize(word) for word in d.split()])
    return [lemmatize(desc) for desc in questions]


def get_vectorizer(questions):
    vect = TfidfVectorizer(max_features= 20000)
    #vect = CountVectorizer(max_df= .6, min_df= 5)
    return vect.fit(questions)

q_train, q_test, y_train, y_test = train_test_split(questions, values, random_state= 42)


vect = get_vectorizer(q_train)
X_train = vect.transform(lemmatize_descriptions(q_train)).toarray()
X_test = vect.transform(lemmatize_descriptions(q_test)).toarray()
'''
text_clf= Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
                    ])

text_clf= text_clf.fit(q_train, y_train)
'''
#get top 5 features
def extract_top_features(feature_matrix):
    matrix= np.argsort(feature_matrix)[:,19997:]
    features= vect.get_feature_names()
    top_terms= []
    for line in matrix:
        top_terms.append([features[i] for i in line])
    return top_terms

def big_words(feature_matrix):
    big_words_index= np.argmax(feature_matrix, axis = 1)
    features= vect.get_feature_names()
    top_terms1= [features[i] for i in big_words_index]
    top_terms= [unidecode(item) for item in top_terms1]
    big_words=[]
    for item in top_terms:
        if len(item) > 15:
            big_words.append(3)
        elif len(item) > 10:
            big_words.append(2)
        else:
            big_words.append(1)
    return np.array(big_words).reshape(len(feature_matrix),1)

def calc_cooccurrence(list_of_features, term_freqs):

#Gets an array of top 5 most important features per question.
#Returns the sum of the co-occurances for this group of features.
    list_of_features= [unidecode(item) for item in list_of_features]
    for word in list_of_features:
        if word not in term_freqs:
            term_freqs[word]= {word: .001}
    freqs= []
    for feature in list_of_features:
        freqs.append(np.array([term_freqs[feature].get(item,.001) for item in list_of_features]))
    normalized= []
    for np_group in freqs:
        normed= np_group/float(np_group.max())
        normalized.append(sum(normed) - 1)
    return sum(normalized)

def important_words(feature_matrix):
    common_index= np.argmax(feature_matrix, axis = 1)
    features= vect.get_feature_names()
    top_terms= [features[i] for i in big_words_index]
    important_words= [unidecode(item) for item in top_terms]
    return important_words


term_freqs_train= dict_of_term_cooccurrence(q_train)
cooccurrence_array= np.array([calc_cooccurrence(row, term_freqs_train) for row in extract_top_features(X_train)])
X_train= np.append(X_train, cooccurrence_array.reshape(len(X_train),1), axis= 1)

#term_freqs_test= dict_of_term_cooccurrence(q_test)
cooccurrence_array2= np.array([calc_cooccurrence(row, term_freqs_train) for row in extract_top_features(X_test)])
X_test= np.nan_to_num(X_test)
X_test= np.append(X_test, cooccurrence_array2.reshape(len(X_test),1), axis= 1)

#X_train = pd.DataFrame(X_train, columns = vect.vocabulary_)
'''
X_train= X_train - np.mean(X_train, axis= 0)
pcd_X= PCA(100).fit_transform(X_train)
X_test= X_test - np.mean(X_test, axis= 0)
pcd_X_test= PCA(100).fit_transform(X_test)

X_train= np.append(X_train, big_words(X_train), axis= 1)
X_test= np.append(X_test, big_words(X_test), axis= 1)
'''
model= MultinomialNB()
model.fit(X_train, y_train)
#scores= cross_val_score(model, X_test, y_test, cv=3)

y_predict= model.predict(X_test)

#y_predict= np.load('predictions.npy').reshape(len(y_test), 1)
#y_test= np.array(y_test).reshape(len(y_test), 1)
print accuracy_score(y_test, y_predict)
'''
y_violin= np.append(y_test, y_predict, axis= 1)
data= [y_violin[y_violin[:,0]== i][:,1] for i in xrange(1,6)]
plt.violinplot(data, range(1,6), showmeans= True)
plt.show()

predicted= text_clf.predict(q_test)
print np.mean(predicted == y_test)
'''
