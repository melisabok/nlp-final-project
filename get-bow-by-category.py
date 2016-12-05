import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk.stem as stem

import csv
from six import iteritems
import os
import argparse

## Argument handling
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', help="the arxiv category to use. example: cs.AI")
args = parser.parse_args()

#Set corpus name. This lets us select from "corpus-titles", "corpus-abstracts", and "corpus-titles-abstracts"
corpus_filename = 'corpus-titles-abstracts'
category = args.category
category_filename = corpus_filename.replace('corpus','category')

if not os.path.exists('./data/%s' % category):
    os.makedirs('./data/%s' % category)

#Set up tokenizer and stop words
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
stops = [word for word in stopwords.words('english')]
stops += ["=", "->", ".", ","]
porter = stem.PorterStemmer()

def tokenize(text):
    return [porter.stem(token) for token in simple_preprocess(text) if token not in STOPWORDS]

#Use a bigram model from JUST the desired category
bigram = Phrases(tokenize(line) for line,label in zip(open('./data/%s.csv' % corpus_filename), open('./data/corpus-labels.csv')) if category in label)

#Make the dictionary, a collection of statistics about tokens in the desired category
#This is the mapping from words to their id's. It's the lookup table for features.
dictionary = corpora.Dictionary(bigram[tokenize(line)] for line,label in zip(open('./data/%s.csv' % corpus_filename), open('./data/corpus-labels.csv')) if category in label)

# find stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stops 
            if stopword in dictionary.token2id]

# remove stop words and words that appear only once
dictionary.filter_tokens(stop_ids)
dictionary.filter_extremes(no_above=0.05, no_below=10) #no_above=0.05, no_below=10 yielded good results
# remove gaps in id sequence after words that were removed
dictionary.compactify()

# store the dictionary, for future reference
dictionary.save('./data/%s/%s.dict' % (category,category_filename))

# memory-friendly bag-of-words class
class BOW(object):
    def __iter__(self):
        for line,label in zip(open('./data/%s.csv' % corpus_filename), open('./data/corpus-labels.csv')):
            # assume there's one document per line, tokens separated by whitespace
            if category in label:
            	yield dictionary.doc2bow(tokenize(line))
            else:
            	pass

# Now we can make a bag of words and do something with it by iterating over it
arxiv_bow = BOW()
corpora.MmCorpus.serialize('./data/%s/%s.mm' % (category,category_filename), arxiv_bow)  # store to disk, for later use

#Create a token to feature ID map. Given a token, gives the feature ID of that token.
token2id_map = dictionary.token2id

# Represent an unseen document as a bag-of-words using this dictionary to define the vector space.
# The function doc2bow() simply counts the number of occurrences of each distinct word, converts 
# the word to its integer word id and returns the result as a sparse vector. The sparse vector 
# [(0, 1), (1, 1)] therefore reads: in the document "all partial results illustrated entropy", the words all (id=31) and partial (id=82) appear once; words that don't appear in the corpus are ignored
print "Represent the following unseen \"document\":\"all partial results results illustrated entropy\""
new_doc = "all partial results results illustrated entropy"
new_vec = dictionary.doc2bow(tokenize(new_doc))
print "Representation: ",new_vec
