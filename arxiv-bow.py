import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk.stem as stem

import csv
from six import iteritems

#Set up tokenizer and stop words
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
stops = [word for word in stopwords.words('english')]
stops += ["=", "->", ".", ","]
porter = stem.PorterStemmer()

def tokenize(text):
    return [porter.stem(token) for token in simple_preprocess(text) if token not in STOPWORDS]

#Make the dictionary, a collection of statistics about all tokens in the corpus
#This is the mapping from words to their id's. It's the lookup table for features.
dictionary = corpora.Dictionary(tokenize(line) for line in open('./data/corpus-abstracts.csv'))

# find stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stops 
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]

# remove stop words and words that appear only once
dictionary.filter_tokens(stop_ids + once_ids)

# remove gaps in id sequence after words that were removed
dictionary.compactify()

# store the dictionary, for future reference
dictionary.save('./data/corpus-abstracts.dict')

# memory-friendly bag-of-words class
class BOW(object):
    def __iter__(self):
        for line in open('./data/corpus-abstracts.csv'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(tokenize(line))

# Now we can make a bag of words and do something with it by iterating over it
arxiv_bow = BOW()
corpora.MmCorpus.serialize('./data/corpus-abstracts.mm', arxiv_bow)  # store to disk, for later use

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
