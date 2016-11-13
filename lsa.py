from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('data/corpus-abstracts.dict')
corpus = corpora.MmCorpus('data/corpus-abstracts.mm')

#Initialize the TF/IDF transformation
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.save('./data/corpus-model.lsi') # same for tfidf, lda, ...

print lsi.print_topics()
