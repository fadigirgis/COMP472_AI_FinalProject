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
from pandas.core.frame import DataFrame


synonyms_data = pd.read_csv('synonyms.csv', delimiter=',')
synonyms_QA = synonyms_data.drop(columns=['question'])
list_of_rows = [list(row) for row in synonyms_QA.values]
# synonyms_QA = pd.DataFrame(synonyms_data, columns=['question','answer'])
print('\n')
print('*****list_of_rows****')
print(list_of_rows)


# synonyms_vec = synonyms_QA.to_numpy()
# list(synonyms_vec)
# print('*****synonyms_vec****')
# print(synonyms_vec)
# print('\n')
synonyms_QA = synonyms_QA.values.tolist()
print('*****synonyms_QA****')
print(synonyms_QA)
print(len(synonyms_QA))

syn_vec = [i[0] for i in list_of_rows]
syn_vec = list(chain.from_iterable(list_of_rows))
print('*****syn_vec*****')
print(syn_vec)

file_AnswerGL50 = synonyms_data
list_of_QA = [list(row) for row in file_AnswerGL50.values]

print('\n')
print('*****list_of_QA****')
print(list_of_QA)


file_AnswerGL50 = synonyms_data.drop(columns=['answer', '0', '1', '2', '3'])
list_of_quest = [list(row) for row in file_AnswerGL50.values]


# Google_results = {"Question Word": [],
#   "Predicted": [], "Synonyme": [], "Result": []}


Glo50_result = []
glov_model50 = api.load("glove-wiki-gigaword-50")
# Glo50_result = {"Question Word": [],
#                 "Predicted": [], "Synonyme": [], "Result": []}


for i in (list_of_quest):
    try:
        Glo50_result.append(glov_model50.most_similar(positive=i, topn=5))
    except:
        Glo50_result.append([('None Existing', 0)])


# list(Glo50_result[list()])
# print('*****Glo50_result****')
print(Glo50_result)


flat_Glov50Result = list(chain.from_iterable(Glo50_result))
# print('\n')
# print('*****flat_Glov50Result****')
# print(flat_Glov50Result)
list_Gl50 = []

for i in flat_Glov50Result:
    list_Gl50.append(i[0])

n = 5
list_Gl50 = [list_Gl50[i:i+n] for i in range(0, len(list_Gl50), n)]
print('\n')
print('*****list_Gl50****')
print(list_Gl50)

# for i in range(len(syn_vec)):
#     if(any(item in list_of_rows for item in list_Gl50)):
#         truth_vectGL50.append('correct')
#     else:
#         truth_vectGL50.append('wrong')

# for m in list_of_rows:
#     for n in list_Gl50:
#         if(m == n):
#             truth_vectGL50.append('correct')
#         else:
#             truth_vectGL50.append('wrong')
guess_Word = []
truth_vectGL50 = []
for i in range(len(list_of_QA)):
    if (any(item in list_of_QA[i] for item in list_Gl50[i])):
        truth_vectGL50.append('correct')
        guess_Word.append(list_Gl50[i])

    else:
        truth_vectGL50.append('wrong')
        guess_Word.append(list_of_rows[i])

# for x in range(len(list_of_quest)):
#     for y in range(len(list_of_QA[x])):
#         if (any(i in list_of_QA[x] for i in list_Gl50[x])):
#             truth_vectGL50.append('correct')
#             guess_Word.append(list_Gl50[x][y])
#         else:
#             truth_vectGL50.append('wrong')
#             guess_Word.append(list_Gl50[x][y])


s1 = set(list_Gl50)
s2 = set(list_of_QA)
guess_Word = s1.intersection(s2)

print('\n')
print('-----------guess_Word-------------')
print(guess_Word)
# for i in range(len(list_of_rows)):
#     for j in range(len(list_of_rows[i])):
#         if (list_of_rows[i][j] in list_Gl50[i]):
#             truth_vectGL50.append('correct')
#             guess_Word.append(list_of_rows[i][j])
#         else:
#             truth_vectGL50.append('wrong')
#             guess_Word.append(list_Gl50[i][j])

# for i in range(len(list_of_rows)):
#     if(list_of_rows[i] in list_Gl50[i]):
#         truth_vectGL50.append('correct')
#     else:
#         truth_vectGL50.append('wrong')


print('\n')
print('-----------truth_vectGL50-------------')
print(truth_vectGL50)


# syn_final = np.intersect1d(syn_1, syn_2)

# for r in range(len(syn_1)):
#     for c in range(len(syn_2)):
#         if(np.intersect1d(syn_2[r], syn_1[c])):
#             truth_vectGL50.append('correct')
#         else:
#             truth_vectGL50.append('wrong')


# for i in range(len(list_Gl50)):
#     syn_vecOf5s = np.intersect1d(syn_1[i], syn_2[i])
#     print('\n')
#     print('-----------syn_vecOf5s-------------')
#     print(syn_vecOf5s)


# if not any(syn_1[i] in x for x in list_Gl50[i]):


# for i in range(len(syn_2)):
#     # if ((np.array(list_Gl50[i]) == np.array(syn_1[i])).all().any()):
#     if (i in syn_2 for i in syn_1):
#         truth_vectGL50.append('correct')
#     else:
#         truth_vectGL50.append('wrong')


# addList = ['None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None',
#            'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None']
# print(len(addList))
# i = 5
# while i < len(syn_1):
#     syn_vec.insert(i, addList)
#     i += 5

# print('\n')
# print('*****SYN_Resized****')
# print(syn_vec)


# for i in range(len(syn_vec)):
#     for j in range(len(syn_vec)):
#         if (syn_vec[i][j] == syn_2[j][i]):
#             truth_vectGL50.append('correct')
#         else:
#             truth_vectGL50.append('wrong')


# print('\n')
# print('*****truth_vectGL50****')
# print(truth_vectGL50)


final_VectorFL50 = [(list_of_rows), (list_Gl50), (list_of_quest)]
print('\n')
print('*****final_VectorFL50****')
print(final_VectorFL50)
csv_vec = list(zip(*final_VectorFL50))


GL50final_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
GL50final_dict['Question'] = list(chain.from_iterable(list_of_quest))
GL50final_dict['Answer'] = list(synonyms_QA)
GL50final_dict['Model Guess'] = guess_Word
GL50final_dict['Label'] = truth_vectGL50


print('\n')
print('****----------*DICT*-------------**')
print('\n')
print('*****list_of_quest****')
print(len(list(chain.from_iterable(list_of_quest))))
print('\n')
print('*****list_of_rows****')
print(len(list(chain.from_iterable(synonyms_QA))))
print('\n')
print('*****guess_Word****')
print(len(guess_Word))
print('\n')
print('*****truth_vectGL50****')
print(len(truth_vectGL50))

print('\n')
print('*****final_dict****')
print(GL50final_dict)


(pd.DataFrame.from_dict(data=GL50final_dict, orient='columns')).to_csv(
    'glove-wiki-gigaword-50.csv', header=True)

# df = (pd.DataFrame.from_dict(GL50final_dict, orient='index'))
# df.transpose()
# df.to_csv('glove-wiki-gigaword-50.csv',
#           index=False, header=True, encoding='Utf-8')

c50 = truth_vectGL50.count('correct')
v50 = len(list_Gl50) - list_Gl50.count('None Existing')
accu50 = c50/v50


Gl50_analysList = ['glove-wiki-gigaword-50',
                   '400000', str(c50), str(v50), str(accu50)]
