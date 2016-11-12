from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess, ClippedCorpus
from gensim.parsing.preprocessing import STOPWORDS

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

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

def evaluate(model, dictionary):
	topics = model[dictionary.doc2bow(tokenize("The standard dynamic programming solution for this problem computes the edit-distance between a pair of strings of total"))]
	
	print "Evaluation topics"
	
	for t in topics:
		print model.show_topic(t[0])
		print "t[1]", t[1]

if __name__ == '__main__':
	dictionary = corpora.Dictionary.load('data/corpus-abstracts.dict')
	corpus = corpora.MmCorpus('data/corpus-abstracts.mm')
	lsi = build_lsa_model(corpus, dictionary, 10)
	lsi.save('./data/corpus-model.lsi')
	lda = build_lda_model(corpus, dictionary, 10)
	lda.save('./data/corpus-model.lda')

	evaluate(lsi, dictionary)
	evaluate(lda, dictionary)