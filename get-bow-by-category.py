import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
from gensim import corpora
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk.stem as stem

import csv
from six import iteritems

import utils

import argparse

## Argument handling
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', help="the arxiv category to use. example: cs.AI")
args = parser.parse_args()

def generate_bow(corpus_filename, category, use_bigrams, no_above, no_below):
    if not os.path.exists('./data/%s' % category):
        os.makedirs('./data/%s' % category)

    tokens = [utils.tokenize(line) for line,label in zip(open('./data/%s.csv' % corpus_filename), open('./data/corpus-labels.csv')) if category in label]
    print 'First token', tokens[1]

    category_filename = corpus_filename.replace('corpus','category')

    #Each category gets its own dictionary and its own corpus, but uses the same bigram model
    #that was computed on all the abstracts
    if use_bigrams:
        if not os.path.exists('./data/%s/bigram.bin' % category):
            bigram = Phrases(utils.tokenize(line) for line,label in zip(open('./data/%s.csv' % corpus_filename), open('./data/corpus-labels.csv')) if category in label)
            Phrases.save(bigram,'./data/%s/bigram.bin' % category)
        else:
            bigram = Phrases.load('./data/%s/bigram.bin')

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
    dictionary.save('./data/%s/%s.dict' % (category,category_filename))

    # memory-friendly bag-of-words class
    class BOW(object):
        def __iter__(self):
            for line,label in zip(open('./data/%s.csv' % corpus_filename), open('./data/corpus-labels.csv')):
                # assume there's one document per line, tokens separated by whitespace
                if category in label:
                    yield dictionary.doc2bow(utils.tokenize(line))
                else:
                    pass

    # Now we can make a bag of words and do something with it by iterating over it
    arxiv_bow = BOW()
    corpora.MmCorpus.serialize('./data/%s/%s.mm' % (category,category_filename), arxiv_bow)  # store to disk, for later use


if __name__ == '__main__':

    #Set corpus name. This lets us select from "corpus-titles", "corpus-abstracts", and "corpus-titles-abstracts"
    corpus_filename = 'corpus-titles-abstracts'
    if args.category:
        generate_bow(corpus_filename, args.category, True, 0.05, 10)
    else:
        for category in utils.get_categories():
            generate_bow(corpus_filename, category, True, 0.05, 10)
