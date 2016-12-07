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

import utils

def generate_bow(corpus_filename, use_bigrams, no_above, no_below):

    tokens = [utils.tokenize(line) for line in open('./data/%s.csv' % corpus_filename)]
    print 'First token', tokens[1]

    if use_bigrams:

        if not os.path.exists('./data/bigram.bin'):
            print "data/bigram.bin doesn't exist. Generating and saving bigram model. This could take a while."
            bigram = Phrases(tokenize(line) for line in open('./data/%s.csv' % corpus_filename))
            bigram.save('./data/bigram.bin')

        bigram = Phrases.load('./data/bigram.bin')
        tokens = [bigram[token] for token in tokens]
        print 'First bigram token', tokens[1]


    #Make the dictionary, a collection of statistics about all tokens in the corpus
    #This is the mapping from words to their id's. It's the lookup table for features.
    dictionary = corpora.Dictionary(tokens)

    # words that appear only once
    dictionary.filter_extremes(no_above, no_below) #no_above=0.05, no_below=10 yielded good results
    # remove gaps in id sequence after words that were removed
    dictionary.compactify()

    # store the dictionary, for future reference
    dictionary.save('./data/%s.dict' % corpus_filename)

    # memory-friendly bag-of-words class
    class BOW(object):
        def __iter__(self):
            for token in tokens:
                # assume there's one document per line, tokens separated by whitespace
                yield dictionary.doc2bow(token)

    # Now we can make a bag of words and do something with it by iterating over it
    arxiv_bow = BOW()
    corpora.MmCorpus.serialize('./data/%s.mm' % corpus_filename, arxiv_bow)  # store to disk, for later use


if __name__ == '__main__':

    #Set corpus name. This lets us select from "corpus-titles", "corpus-abstracts", and "corpus-titles-abstracts"
    corpus_filename = 'corpus-titles-abstracts'
    generate_bow(corpus_filename, True, 0.05, 10)
