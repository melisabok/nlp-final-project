import math
import csv
from collections import Counter
from collections import defaultdict
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess, ClippedCorpus
from gensim.parsing.preprocessing import STOPWORDS
import nltk.stem as stem

porter = stem.PorterStemmer()

def tokenize(text):
    return [porter.stem(token) for token in simple_preprocess(text) if token not in STOPWORDS]

lda_all = models.LdaModel.load('./data/corpus-titles-abstracts.lda')

dictionary_all = corpora.Dictionary.load('data/corpus-titles-abstracts.dict')

sampletext = "On Classification from Outlier View Classification is the basis of cognition. Unlike other solutions, this study approaches it from the view of outliers. We present an expanding algorithm to detect outliers in univariate datasets, together with the underlying foundation. The expanding algorithm runs in a holistic way, making it a rather robust solution. Synthetic and real data experiments show its power. Furthermore, an application for multi-class problems leads to the introduction of the oscillator algorithm. The corresponding result implies the potential wide use of the expanding algorithm."

target_vec = lda_all.get_document_topics(dictionary_all.doc2bow(tokenize(sampletext)), per_word_topics=True)[0]

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


closest_points = []
count = 1;
k = 100
with open('./data/corpus-titles-abstracts.csv') as corpus, open('./data/corpus-labels.csv') as labels:
    corpusreader = csv.reader(corpus)
    labelsreader = csv.reader(labels)
    
    for l,c in zip(labelsreader,corpusreader):      
        current_vec = lda_all[dictionary_all.doc2bow(tokenize(c[0]))]
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
print category_counter