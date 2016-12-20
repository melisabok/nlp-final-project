from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess, ClippedCorpus
from gensim.parsing.preprocessing import STOPWORDS
import nltk.stem as stem
import pickle

porter = stem.PorterStemmer()


def tokenize(text):
    return [porter.stem(token) for token in simple_preprocess(text) if token not in STOPWORDS]

def unstem(text,keyword_stems):
    words = [token for token in simple_preprocess(text) if token not in STOPWORDS]
    keywords = set()

    for k in keyword_stems:
        word = next((w for w in words if k in w), None)
        if word:
            keywords.add(word)
    return keywords

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_categories():
    return {
        "cs.AR":"Computer Science - Architecture",
        "cs.AI":"Computer Science - Artificial Intelligence",
        "cs.CL":"Computer Science - Computation and Language",
        "cs.CC":"Computer Science - Computational Complexity",
        "cs.CE":"Computer Science - Computational Engineering; Finance; and Science",
        "cs.CG":"Computer Science - Computational Geometry",
        "cs.GT":"Computer Science - Computer Science and Game Theory",
        "cs.CV":"Computer Science - Computer Vision and Pattern Recognition",
        "cs.CY":"Computer Science - Computers and Society",
        "cs.CR":"Computer Science - Cryptography and Security",
        "cs.DS":"Computer Science - Data Structures and Algorithms",
        "cs.DB":"Computer Science - Databases",
        "cs.DL":"Computer Science - Digital Libraries",
        "cs.DM":"Computer Science - Discrete Mathematics",
        "cs.DC":"Computer Science - Distributed; Parallel; and Cluster Computing",
        "cs.GL":"Computer Science - General Literature",
        "cs.GR":"Computer Science - Graphics",
        "cs.HC":"Computer Science - Human-Computer Interaction",
        "cs.IR":"Computer Science - Information Retrieval",
        "cs.IT":"Computer Science - Information Theory",
        "cs.LG":"Computer Science - Learning",
        "cs.LO":"Computer Science - Logic in Computer Science",
        "cs.MS":"Computer Science - Mathematical Software",
        "cs.MA":"Computer Science - Multiagent Systems",
        "cs.MM":"Computer Science - Multimedia",
        "cs.NI":"Computer Science - Networking and Internet Architecture",
        "cs.NE":"Computer Science - Neural and Evolutionary Computing",
        "cs.NA":"Computer Science - Numerical Analysis",
        "cs.OS":"Computer Science - Operating Systems",
        "cs.OH":"Computer Science - Other",
        "cs.PF":"Computer Science - Performance",
        "cs.PL":"Computer Science - Programming Languages",
        "cs.RO":"Computer Science - Robotics",
        "cs.SE":"Computer Science - Software Engineering",
        "cs.SD":"Computer Science - Sound",
        "cs.SC":"Computer Science - Symbolic Computation"
    }
