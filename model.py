import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts

dataset = api.load('word2vec-google-news-300')

print('hi')