from logging import exception
from sys import excepthook
from gensim import models
import csv
import operator
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec, word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts
from numpy import negative, positive
from itertools import chain


synonyms_data = pd.read_csv('synonyms.csv')
# synonyms_QA = pd.DataFrame(synonyms_data, columns=['question','answer'])
synonyms_QA = synonyms_data.drop(columns=['0', '1', '2', '3'])
synonyms_vec = synonyms_QA.to_numpy()

print('*****synonyms_vec****')
print(synonyms_vec)
print('\n')

print('*****synonyms_QA****')
print(synonyms_QA)
# model1 = Word2Vec(synonyms_vec)
# print(model1)
# Google_model = Word2Vec (questionword_vec, min_count=1)

syn_vec = [i[0] for i in synonyms_vec]
syn_vec = np.reshape(syn_vec, -1)
print('*****syn_vec*****')
print(syn_vec)
Google_results = {"Question Word": [],
                  "Predicted": [], "Synonyme": [], "Result": []}
Google_result = []
# Google_result = np.reshape(Google_result,-1)
Google_model = api.load('word2vec-google-news-300')
# Google_model = word2vec.Word2Vec(Google_dataset)
for i in (syn_vec):
    try:
        Google_result.append(Google_model.most_similar(positive=i, topn=1))
    # print(Google_result)

        # Google_result.append[[(Google_model.most_similar(positive= i, topn = 1))]]
    except:
        Google_result.append([('None Existing', 0)])


print('*****Google_result****')
print(Google_result)


flat_GooResult = list(chain.from_iterable(Google_result))
print('\n')
print('*****flat_GooResult****')
print(flat_GooResult)
first_GooSyn = []

for i in flat_GooResult:
    first_GooSyn.append(i[0])


print('\n')
print('*****first_GooSyn****')
print(first_GooSyn)


file_Answer = synonyms_data.drop(columns=['question', '0', '1', '2', '3'])
print('\n')
print('*****file_Answer****')
print(file_Answer)
file_answerVec = file_Answer.to_numpy()
print('\n')
print('*****file_answerVec****')
print(file_answerVec)


truth_vect = []
for i in range(len(syn_vec)):
    if(file_answerVec[i] == first_GooSyn[i]):
        truth_vect.append('correct')
    else:
        truth_vect.append('wrong')

print('\n')
print('*****truth_vect****')
print(truth_vect)

final_Vector = [(synonyms_vec), (first_GooSyn), (truth_vect)]
print('\n')
print('*****final_Vector****')
print(final_Vector)
csv_vec = list(zip(*final_Vector))


final_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
final_dict['Question'] = syn_vec.tolist()
final_dict['Answer'] = list(chain.from_iterable(file_answerVec))
final_dict['Model Guess'] = first_GooSyn
final_dict['Label'] = truth_vect
print('\n')
print('*****final_dict****')
print(final_dict)

# with open('word2vec-google-news-300-details.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     for key, value in final_dict.items():
#         writer.writerow([key, value])

# csvfile.close()

(pd.DataFrame.from_dict(data=final_dict, orient='columns')).to_csv(
    'word2vec-google-news-300-details.csv', header=True)

c = truth_vect.count('correct')
v = len(first_GooSyn) - first_GooSyn.count('None Existing')
accu = c/v


Goog_analysList = ['word2vec-google-news-300',
                   '3000000', str(c), str(v), str(accu)]

with open('analysis.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    # for key, value in analysis_dict.items():
    writer.writerow(Goog_analysList)


# model = api.load("glove-wiki-gigaword-50")
# model.most_similar("glass")
# most_similar_key, similarity = Google_result[0]
# print(f"{most_similar_key}: {similarity:.4f}")from gensim.scripts.glove2word2vec import glove2word2vec
# glove_input_file = 'glove.txt'
# word2vec_output_file = 'word2vec.txt'
# glove2word2vec(glove_input_file, word2vec_output_file)
