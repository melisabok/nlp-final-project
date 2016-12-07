from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess, ClippedCorpus
from gensim.parsing.preprocessing import STOPWORDS
import nltk.stem as stem
import argparse
import utils

## Argument handling
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', help="the arxiv category to use. example: cs.AI")
args = parser.parse_args()

porter = stem.PorterStemmer()

def tokenize(text):
    return [porter.stem(token) for token in simple_preprocess(text) if token not in STOPWORDS]

def build_lsa_model(corpus, dictionary, num_topics):

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)

    print "LSA Topics:"
    for i in lsi.show_topics():
        print "Topic: ", i[0], i[1]

    return lsi

def build_lda_model(corpus, dictionary, num_topics):

    clipped_corpus = ClippedCorpus(corpus, 4000)
    lda = models.LdaModel(clipped_corpus, id2word=dictionary, num_topics=num_topics, passes=4)

    print "LDA topics"
    print lda.show_topics()

    return lda

def evaluate(model, dictionary,text):
    topics = model[dictionary.doc2bow(tokenize(text))]

    print "Evaluation topics"

    for t in topics:
        print model.show_topic(t[0])
        print "t[1]", t[1]

if __name__ == '__main__':

    corpus_filename = 'corpus-titles-abstracts'
    if args.category:
        category = args.category
        category_filename = corpus_filename.replace('corpus','category')

        dictionary = corpora.Dictionary.load('data/%s/%s.dict' % (category,category_filename))
        corpus = corpora.MmCorpus('data/%s/%s.mm' % (category,category_filename))
        lsi = build_lsa_model(corpus, dictionary, 10)
        lsi.save('./data/%s/%s.lsi' % (category,category_filename))
        lda = build_lda_model(corpus, dictionary, 10)
        lda.save('./data/%s/%s.lda' % (category,category_filename))
    else:
        for category in utils.get_categories():
            category_filename = corpus_filename.replace('corpus','category')

            dictionary = corpora.Dictionary.load('data/%s/%s.dict' % (category,category_filename))
            corpus = corpora.MmCorpus('data/%s/%s.mm' % (category,category_filename))
            lsi = build_lsa_model(corpus, dictionary, 10)
            lsi.save('./data/%s/%s.lsi' % (category,category_filename))
            lda = build_lda_model(corpus, dictionary, 10)
            lda.save('./data/%s/%s.lda' % (category,category_filename))
