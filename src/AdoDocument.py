from nltk.corpus import stopwords
from nltk import download
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
import nltk
import os

nltk.download('averaged_perceptron_tagger')
# Load stopwords and stemmers
download('stopwords')
engStopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
nltk.data.path.append(os.environ['NLTK_DIR'])

separator = '|'

'''
    Holder class for a document in the corpus
'''


class AdoDocument:

    def __init__(self, id, hashtagsString, tokensString):
        self.id = id  # Document ID
        self.hashtags = []  # List of hashtags
        self.tokens = []  # List of words
        self.text =''
        self.add(hashtagsString, tokensString)

    def add(self, hashtagsString, tokensString):
        self.hashtags += self.__returnOnlyNotNull(hashtagsString)
        self.tokens += self.__returnOnlyNotNull(tokensString)
        self.text= ' '.join(self.tokens)

    def print(self):
        print(f'id:{self.id}\n hashtags:{separator.join(self.hashtags)}\n tokens:{separator.join(self.tokens)}')

    # Convert tokens to terms by: removing non-nouns, non-alphabetic words, and stopwords, then stemming
    def tokens2terms(self):
        self.text= self.tokens
        terms = [x[0] for x in pos_tag(self.tokens) if x[1][0:1] == 'N']
        terms = [x for x in terms if x.isalpha()]
        terms = [x for x in terms if x not in engStopwords]
        terms = [stemmer.stem(x) for x in terms]
        self.tokens = terms

    def __returnOnlyNotNull(self, s):
        return [t for t in s.split(separator) if len(t) > 0]
