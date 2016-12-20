
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess, ClippedCorpus
from gensim.parsing.preprocessing import STOPWORDS
import nltk.stem as stem
import utils


def build_lda_model(corpus, dictionary, num_topics):

    #clipped_corpus = ClippedCorpus(corpus, 4000)
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)

    print "LDA topics: ", num_topics
    #print lda.show_topics()
    print "Perplexity", lda.log_perplexity(corpus)

    return lda

def evaluate(model, dictionary, doc):

    words = []
    for w in model.get_document_topics(dictionary.doc2bow(utils.tokenize(doc)), per_word_topics=True)[2]:
        words.append((dictionary[w[0]], w[1][0][1]))

    best_words = sorted(words, key=lambda tup: tup[1], reverse=True)[:10]
    best_words = [t[0] for t in best_words]

    print "Evaluation topics"
    print utils.unstem(doc, best_words)

if __name__ == '__main__':

    corpus_name = 'corpus-titles-abstracts'
    dictionary = corpora.Dictionary.load('data/%s.dict' % corpus_name)
    corpus = corpora.MmCorpus('data/%s.mm' % corpus_name)


    for num_topics in range(5, 256, 10):

        lda = build_lda_model(corpus, dictionary, num_topics)
        #lda.save('./data/%s.lda' % corpus_name)

        doc = """Optimum Linear LLR Calculation for Iterative Decoding on Fading Channels On a fading channel with no channel state information at the receiver,
        calculating true log-likelihood ratios (LLR) is complicated. Existing work assume that
        the power of the additive noise is known and use the expected value of the fading gain
        in a linear function of the channel output to find approximate LLRs. In this work,
        we first assume that the power of the additive noise is known and we find the optimum
        linear approximation of LLRs in the sense of maximum achievable transmission rate on the channel.
        The maximum achievable rate under this linear LLR calculation is almost equal to the maximum achievable
        rate under true LLR calculation. We also observe that this method appears to be the optimum in the sense
        of bit error rate performance too. These results are then extended to the case that the noise power
        is unknown at the receiver and a performance almost identical to the case that the noise power
        is perfectly known is obtained."""

        evaluate(lda, dictionary, doc)
