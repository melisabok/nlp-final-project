import math
import csv
from collections import Counter
from collections import defaultdict
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess, ClippedCorpus
from gensim.parsing.preprocessing import STOPWORDS
import nltk.stem as stem
import utils
import re
import os

categories = utils.get_categories()

category_models = {category:models.LdaModel.load('./data/%s/category-titles-abstracts.lda' % category) for category in categories}
category_dicts = {category:corpora.Dictionary.load('data/%s/category-titles-abstracts.dict' % category) for category in categories}
category_bigrams = {category:models.Phrases.load('data/%s/bigram.bin' % category) for category in categories}

lda_all = models.LdaModel.load('./data/corpus-titles-abstracts.lda')
dictionary_all = corpora.Dictionary.load('data/corpus-titles-abstracts.dict')

bigram_all = models.Phrases.load('./data/bigram.bin')

sampletext = "On Classification from Outlier View Classification is the basis of cognition. Unlike other solutions, this study approaches it from the view of outliers. We present an expanding algorithm to detect outliers in univariate datasets, together with the underlying foundation. The expanding algorithm runs in a holistic way, making it a rather robust solution. Synthetic and real data experiments show its power. Furthermore, an application for multi-class problems leads to the introduction of the oscillator algorithm. The corresponding result implies the potential wide use of the expanding algorithm."

def save_pointcloud(filepath):
    ldaspace = []
    with open('./data/corpus-titles-abstracts.csv') as corpus, open('./data/corpus-labels.csv') as labels:
        corpusreader = csv.reader(corpus)
        labelsreader = csv.reader(labels)
        print 'starting'

        for l,c in zip(labelsreader,corpusreader):
            current_vec = lda_all[dictionary_all.doc2bow(utils.tokenize(c[0]))]
            ldaspace.append(current_vec)

    utils.save_obj(ldaspace,filepath)


def get_distance(current_vec,target_vec):
    p1 = defaultdict(lambda:0.0)
    p2 = defaultdict(lambda:0.0)

    for d in current_vec:
        p1[d[0]] = d[1]
    for d in target_vec:
        p2[d[0]] = d[1]

    terms = []
    for d in range(10):
        terms.append((p1[d]-p2[d])**2)
    return math.sqrt(sum(terms))



def classify(k,text):

    target_vec = lda_all.get_document_topics(dictionary_all.doc2bow(utils.tokenize(sampletext)), per_word_topics=True)[0]

    closest_points = []

    with open('./data/corpus-labels.csv') as labels:
        labelreader = csv.reader(labels)

        if not os.path.exists('./data/ldaspace-titles-abstracts.pkl'):
            print "data/ldaspace-titles-abstract.pkl not found. Generating file (this may take a while)"
            save_pointcloud('./data/ldaspace-titles-abstracts')

        ldaspace = utils.load_obj('./data/ldaspace-titles-abstracts')

        for l,current_vec in zip(labelreader,ldaspace):

            dist = get_distance(current_vec,target_vec)
            if len(closest_points) >= k:
                if dist < closest_points[k-1]:
                    closest_points.pop(k-1)
                    closest_points.append((l,dist))
            else:
                closest_points.append((l,dist))

            closest_points.sort(key=lambda point: point[1])

    category_counter = Counter()
    for x in closest_points:
        category_counter.update(x[0])

    return category_counter

def get_knn_keywords(k=100, top_n=3, text=sampletext, counts=None):

    if not counts:
        counts = classify(k,text)

    category_lists = defaultdict(list)
    for k,v in Counter({label:counts[label] for label in counts if label in utils.get_categories()}).most_common(top_n):
        lda_cat = category_models[k]
        dictionary_cat = category_dicts[k]
        bigram_cat = category_bigrams[k]

        for w in lda_cat.get_document_topics(dictionary_cat.doc2bow(bigram_cat[utils.tokenize(text)]), per_word_topics=True)[2]:
            category_lists[k].append(dictionary_cat[w[0]])

    for w in lda_all.get_document_topics(dictionary_all.doc2bow(bigram_all[utils.tokenize(text)]), per_word_topics=True)[2]:
        category_lists['all'].append(dictionary_all[w[0]])

    return category_lists

if __name__ == '__main__':
    category_counter = knn()
    print category_counter
