import json
import os
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from collections import OrderedDict
from nltk.corpus import stopwords
import numpy as np
import math
import itertools
import re

#defining stopwords 
_WORD_MIN_LENGTH = 3
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
_STOP_WORDS = stopwords.words('english') + list(punctuation) + ['\n']


#splitting a text and storing in into a new list 
def word_split(text):
    word_list = []
    wcurrent = []
    windex = None

    for i, c in enumerate(text):
        if c.isalnum():
            wcurrent.append(c)
            windex = i
        elif wcurrent:
            word = u''.join(wcurrent)
            word_list.append((windex - len(word) + 1, word))
            wcurrent = []

    if wcurrent:
        word = u''.join(wcurrent)
        word_list.append((windex - len(word) + 1, word))

    return word_list


#Cleaning the words by removing stopwords and having minimum length
def words_cleanup(words):
    cleaned_words = []
    for index, word in words:
        if len(word) < _WORD_MIN_LENGTH or word in _STOP_WORDS:
            continue
        cleaned_words.append((index, word))
    return cleaned_words


#normalising the words by changing the all words in dictonary into lower-case letters
def words_normalize(words):
    normalized_words = []
    for index, word in words:
        wnormalized = word.lower()
        normalized_words.append((index, wnormalized))
    return normalized_words



def word_index(text):
    words = word_split(text)
    words = words_normalize(words)
    words = words_cleanup(words)
    return words



def inverted_index(text):
    inverted = {}
    for index, word in word_index(text):
        locations = inverted.setdefault(word, [])
        locations.append(index)
    return inverted



def inverted_index_add(inverted, doc_id, doc_index):
    for word, locations in doc_index.items():
        indices = inverted.setdefault(word, {})
        indices[doc_id] = locations
    return inverted


#creating an inverted index for all the words in the documents
def req_inv_index(inverted):
    for doc_id, items in inverted.items():
        for k in items:
            items[k] = len(items[k])

    return inverted

#creating the json file and storing the dictionary index in the json file
directory = 'Dataset'
documents = {}
inverted = {}

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        infile = open(f, encoding="utf-8")
        words = infile.readlines()
        new = os.path.splitext(filename)[0]
        filename = new
        
        documents[filename] = words[0]+" "+words[1]

docs_length = len(documents) 
# print(docs_length)

for doc_id, text in documents.items():
    doc_idx = inverted_index(text)
    inverted_index_add(inverted, doc_id, doc_idx)

s=req_inv_index(inverted)

with open("indexing.json", "w") as outfile:
    json.dump(s, outfile,indent=4)

    
data = dict()

with open('indexing.json', 'r') as f:
    data = json.load(f)
    f.close()


#calculating the term frequency for all the words in the dictonary
def term_frequency(word , doc_id):
   return 1 + math.log2(data[word][str(doc_id)])

#calculating the documents count for all the documents in the dictonary
documents_count = (len([name 
for name in os.listdir('.\Dataset') 
    if os.path.isfile(os.path.join('.\Dataset', name))]))

#calculating the idf values for the dictionary 
def idf(data,documents_count):
   idf={}
   count = 0
   for key, value in data.items():
       for i in value.keys():
         count = count + 1
       idf[key] = np.log10((documents_count)/count)
       count = 0
   return idf

idf = idf(data,documents_count)
with open('idf.json', 'w') as f:
        json.dump(idf, f, indent=4)
        f.close()

#calculating the idf values for the dictionary 
def tf_idf_values(data,idf):
    tf_idf={}

    for key,value in data.items():
        tf_idf[key]={}
        for i,j in value.items():
            s1=i
            tf_idf[key][s1]= idf[key]*j
    with open('tf_idf.json', 'w') as f:
        json.dump(tf_idf, f, indent=4)
        f.close()
    return tf_idf

tf_idf=tf_idf_values(data,idf)


#Creating the wordvector of the dictionary
def word_vec(idf):
    word_vector = {}
    flag=0
    for i in idf.keys():
        word_vector[i] = flag
        flag = flag+1
    return word_vector 


word_vector= word_vec(idf)

with open('word_vector.json', 'w') as f:
        json.dump(word_vector, f, indent=4)
        f.close()

doc_vectors = dict()
os.chdir('./Dataset')
def vectorizer( doc_id):
    file = open('{}.txt'.format(str(doc_id)), 'r', encoding = 'utf8')
    text = file.readlines()
    document = word_index(text[0]+ text[1])
    words = []
    for pair in document:
        words.append(pair[1])
    # print(word_vector)
    vector = list()
    for word in word_vector:
        if word in words:
            vector.append(tf_idf[word][str(doc_id)])
        else:
            vector.append(0)
    doc_vectors[str(doc_id)] = vector



for i in range(1, documents_count+1):
    vectorizer(i)

os.chdir('..')
with open('document_vector.json', 'w') as f:
        json.dump(doc_vectors, f, indent=4)
        f.close()
