import math 
import numpy as np
from collections import defaultdict
from array import array
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import collections
import pickle
import pandas as pd
nltk.download('stopwords')
#preprocess
import random
import os

from myapp.search.objects import ResultItem, Document

# import the build term function from preprocessing.py

def preprocess(full_text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    full_text=  full_text.lower()
    # rename all urls
    full_text = re.sub(url_pattern, '', full_text)
    # remove all words that start with #
    full_text = re.sub(r'#\w+', '', full_text)
    return full_text
    
    #build_terms
def build_terms(line, tokenize=True):

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    line = preprocess(line)

    line = nltk.tokenize.word_tokenize(line)  # Tokenize the text to get a list of terms
    # here we have to decide if remove # or not and explain why
    line = [word for word in line if re.match(r'^\w+$', word)]  # remove punctuation
    line = [word for word in line if word not in stop_words]  # eliminate the stopwords
    line = [nltk.stem.SnowballStemmer('english').stem(word) for word in line]  # perform stemming

    if tokenize:
        return line
    else:
        return ' '.join(line)


def search_docs_intersection(query,index):
    query = build_terms(query)
    docs = set()
    #initialize
    #obtain the docs that contain the first term
    try:
        docs=set([posting[0] for posting in index[query[0]]])
    except KeyError :
        pass
    
    for term in query[1:]:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs=[posting[0] for posting in index[term]]
            #inersection to have all docs that contain all the terms
            docs = docs.intersection(term_docs)
        except KeyError:
            #term is not in index
            pass
    docs = list(docs)

    return docs

def boolean_search(query, document_index):
    query_terms = query.split()

    docs = set()
    #initialize
    #obtain the docs that contain the first term
    try:
        docs=set([posting[0] for posting in document_index[build_terms(query[0])[0]]])
    except KeyError :
        pass
   
    bolean_operator = ''
    # Process each term in the query
    for term in query_terms:
        
        # Check if the term is a boolean operator (AND, OR, NOT)
        if term.upper() == 'AND':
            bolean_operator = 'AND'
        elif term.upper() == 'OR':
            bolean_operator = 'OR'
        elif term.upper() == 'NOT':
            bolean_operator = 'NOT'
        else:
            # Individual term
            
            try:
                term_set = set([posting[0] for posting in document_index[build_terms(term)[0]]])
                
                if bolean_operator == 'AND':
                    docs = docs.intersection(term_set)
                elif bolean_operator == 'OR':
                    docs = docs.union(term_set)
                elif bolean_operator == 'NOT':
                    docs = docs.difference(term_set)
                else:
                    # If no previous operator, assume OR by default
                    docs = docs.union(term_set)
            except KeyError:
                #term is not in index
                pass
   
    return docs


docs_path = 'Rus_Ukr_war_data_ids.csv'
with open(docs_path) as fp:
    lines = fp.readlines()
doc_num_id = [l.replace("\t", ' ').strip() for l in lines]
# create dictionaries to obtain the tweet_id from the document number and viceversa
id_to_doc_num=dict()
doc_num_to_id=dict()
for i in range(len(doc_num_id)):
    id_to_doc_num[int(doc_num_id[i].split(' ')[1])]=doc_num_id[i].split(' ')[0]
    doc_num_to_id[doc_num_id[i].split(' ')[0]]=int(doc_num_id[i].split(' ')[1])

def get_index_idf_tf():
    # 1. create create_tfidf_index
        #obtain index,idf,df with piclke
    with open('index.pickle', 'rb') as handle:
        index = pickle.load(handle)
    with open('idf.pickle', 'rb') as handle:
        idf = pickle.load(handle)
    with open('tf.pickle', 'rb') as handle:
        tf = pickle.load(handle)
    return index,idf,tf


#2. apply ranking 
    
def rank_documents_our_new_score(corpus,terms, docs, index, idf, tf,tweet_info,delta=0.5):
    # The remaining elements would became 0 when multiplied to the query_vector
    # I'm interested only on the element of the docVector corresponding to the query terms
    doc_vectors = defaultdict(lambda: [0] * (len(terms)+1)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * (len(terms)+1)

    likes=[int(corpus[doc_num_to_id[i]].likes) for i in docs]
    retwets=[int(tweet_info[doc_num_to_id[i]].retweets) for i in docs]

     # normalize the likes and retwets with z-score normalization
    likes_norm = [(like - np.mean(likes)) / np.std(likes) for like in likes]
    retwets_norm = [(retwet - np.mean(retwets)) / np.std(retwets) for retwet in retwets]

     # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue
        ## Compute tf*idf(normalize TF as done with documents)
        # tf_mean = np.mean([val for val in tf[term]])
        # tf_idf_mean = tf_mean * idf[term]
        query_vector[termIndex]=delta*(tf[term][termIndex] * idf[term])

         # Generate doc_vectors for matching docs
        for doc_index, (doc, _) in enumerate(index[term]):

             #tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs:
                doc_vectors[doc][termIndex] = delta*(tf[term][doc_index] * idf[term])

     # set extra dimension for popularity in docs
    for doc in docs:
        current_like=(float(corpus[doc_num_to_id[doc]].likes) - np.mean(likes)) / np.std(likes)
        current_rt=(float(corpus[doc_num_to_id[doc]].retweets) - np.mean(retwets)) / np.std(retwets)
        current_like=(current_like - min(likes_norm)) / (max(likes_norm) - min(likes_norm))
        current_rt=(current_rt - min(retwets_norm)) / (max(retwets_norm) - min(retwets_norm))
        doc_vectors[doc][len(terms)] = (1-delta)*(current_like + current_rt)/2

     # set extra dimension for popularity in query
    query_vector[len(terms)] = (1-delta)



    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    #print document titles instead if document id's
    #result_docs=[ title_index[x] for x in result_docs ]
    if len(result_docs) == 0:
        print('aqui')
        return list()
        
    #print ('\n'.join(result_docs), '\n')
    return result_docs


def search_in_corpus(corpus,query):
    index, idf, tf = get_index_idf_tf()
    docs = search_docs_intersection(query,index)
    if len(docs) == 0:
        return False
    query = build_terms(query)
    ranked_docs =rank_documents_our_new_score(corpus,query, docs, index, idf, tf, corpus,0.7)
   
    ranked_tweetid = [doc_num_to_id[docid] for docid in ranked_docs]
    ranked_tweets = [corpus[tweet_id] for tweet_id in ranked_tweetid]
    return ranked_tweets

def search_in_corpus_idf_itf(corpus,query):
    index, idf, tf = get_index_idf_tf()
    docs = search_docs_intersection(query,index)
    if len(docs) == 0:
        return False
    query = build_terms(query)
    ranked_docs =rank_documents_our_new_score(corpus,query, docs, index, idf, tf, corpus,1)
    ranked_tweetid = [doc_num_to_id[docid] for docid in ranked_docs]
    ranked_tweets = [corpus[tweet_id] for tweet_id in ranked_tweetid]
    return ranked_tweets


def search_in_bolean(corpus,query):
    index, idf, tf = get_index_idf_tf()
    docs = boolean_search(query,index)
    if len(docs) == 0:
        return False
    ranked_tweetid = [doc_num_to_id[docid] for docid in docs]
    ranked_tweets = [corpus[tweet_id] for tweet_id in ranked_tweetid]
    return ranked_tweets