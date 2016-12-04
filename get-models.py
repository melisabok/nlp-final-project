from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess, ClippedCorpus
from gensim.parsing.preprocessing import STOPWORDS
import nltk.stem as stem

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
    
    corpus_name = 'corpus-titles-abstracts'
    dictionary = corpora.Dictionary.load('data/%s.dict' % corpus_name)
    corpus = corpora.MmCorpus('data/%s.mm' % corpus_name)
    lsi = build_lsa_model(corpus, dictionary, 10)
    lsi.save('./data/%s.lsi' % corpus_name)
    lda = build_lda_model(corpus, dictionary, 10)
    lda.save('./data/%s.lda' % corpus_name)
    