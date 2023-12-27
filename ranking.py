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


f = open("idf.json", "r")
idf=json.load(f)
f.close()


f = open("word_vector.json", "r")
word_vector=json.load(f)
f.close()



f = open("document_vector.json", "r")
doc_vectors=json.load(f)
f.close()


query_number = input("please enter the query number: ")
query = input("please enter the query: ")

#Creating the query vector of the words in the dictionary
def query_vec(query,idf):
    query_vocab_stripped = list(set([x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in query.lower().split()]))
    query_list = [x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in query.lower().split()]
    # print(query_list)
    query_wc = {}
    for word in query_vocab_stripped:
        # print(query_list.count(word))
        query_wc[word] = query_list.count(word)
    
    for key,value in query_wc.items():
        # print(type(idf[0][key]),type(value))
        query_wc[key] = value*idf[key]

    query_vector=np.zeros(len(idf))
    for i,j in idf.items():
        if i in query_wc.keys():
            query_vector[word_vector[i]]=query_wc[i]  
    return query_vector





f = open("indexing.json", "r")
tf_idf=json.load(f)
f.close()

f = open("query_relevance.json", "r")
query_relevance=json.load(f)
f.close()

    
os.chdir('./Dataset')


def cos_sim(a, b):
    #norm_a and norm_b would calculate the mod of the vector
    norm_a = np.linalg.norm(a) 
    norm_b = np.linalg.norm(b)
    if (norm_a == 0) or (norm_b == 0):
        sim = 0
    else:
        sim=np.dot(a, b) / ( norm_a * norm_b )
    return sim 




def mat_score(doc_vector,query_vector):
    matching_score={}
    for key,value in doc_vector.items():
       
        if query in query_relevance.keys():
            if key in list(query_relevance[query]):
                 matching_score[key] = cos_sim(query_vector,value)+0.15
            
            else:
              matching_score[key] = cos_sim(query_vector,value)  
        
        else:
            matching_score[key] = cos_sim(query_vector,value)
    
        ranked_docs={}
        ranked_docs = {k: v for k, v in sorted(matching_score.items(), key=lambda item: item[1],reverse=True)}
    return ranked_docs




qv=query_vec(query,idf)

matching_score = mat_score(doc_vectors,qv)


def top_k_documents(k):
  sample = {}
  try:
        sample = dict(itertools.islice(matching_score.items(), k)) 
  except:
        sample = dict(itertools.islice(matching_score.items(), 5)) 
  return sample

    
top_k_final = top_k_documents(10)

os.chdir('..')
ret = {}
if query_number != "1":
        with open("final_ranking.json") as infile:
            ret = json.load(infile)
            ret[query_number] = top_k_final
            ret = json.dumps(ret, indent=4)
        with open("final_ranking.json", "w") as outfile:
            outfile.write(ret)
else:
        with open("final_ranking.json", "w") as outfile:
            ret[query_number] = top_k_final
            outfile.write(json.dumps(ret, indent=4))
ret ={}
if query_number != "1":
        with open("final_quer_ranking.json") as infile:
            ret = json.load(infile)
            ret[query] = top_k_final
            ret = json.dumps(ret, indent=4)
        with open("final_quer_ranking.json", "w") as outfile:
            outfile.write(ret)
else:
        with open("final_quer_ranking.json", "w") as outfile:
            ret[query] = top_k_final
            outfile.write(json.dumps(ret, indent=4))



relevant_docs = {}
if query_number != "1":
        with open("relevance.json") as infile:
            relevant_docs = json.load(infile)
else:
        relevant_docs = {}

query_relevant_docs = {}
if query_number != "1":
        with open("query_relevance.json") as infile:
            query_relevant_docs = json.load(infile)
else:
        query_relevant_docs = {}

relevant_list = []

for i in top_k_final:
        relevance = int(input("Is {doc} relevant to your query? if yes type 1, if no type 0: ".format(doc = i)))
        if relevance == 1:
            relevant_list.append(i)

relevant_docs[query_number] = relevant_list
query_relevant_docs[query]=relevant_list

with open("relevance.json", "w") as outfile:
        relevant = json.dumps(relevant_docs, indent=4)
        outfile.write(relevant) 


with open("query_relevance.json", "w") as outfile:
        relevant = json.dumps(query_relevant_docs, indent=4)
        outfile.write(relevant) 

