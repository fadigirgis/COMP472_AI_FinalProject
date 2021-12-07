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


# ******************************************************************************************
# ******************************Google_model TASK 1*****************************************
# ******************************************************************************************


synonyms_data = pd.read_csv('synonyms.csv')
# synonyms_QA = pd.DataFrame(synonyms_data, columns=['question','answer'])
synonyms_QA = synonyms_data.drop(columns=['0', '1', '2', '3'])
synonyms_vec = synonyms_QA.to_numpy()

# print('*****synonyms_vec****')
# print(synonyms_vec)
# print('\n')

# print('*****synonyms_QA****')
# print(synonyms_QA)


syn_vec = [i[0] for i in synonyms_vec]
syn_vec = np.reshape(syn_vec, -1)
# print('*****syn_vec*****')
# print(syn_vec)
# Google_results = {"Question Word": [],
#   "Predicted": [], "Synonyme": [], "Result": []}
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


# print('*****Google_result****')
# print(Google_result)


flat_GooResult = list(chain.from_iterable(Google_result))
# print('\n')
# print('*****flat_GooResult****')
# print(flat_GooResult)
first_GooSyn = []

for i in flat_GooResult:
    first_GooSyn.append(i[0])


# print('\n')
# print('*****first_GooSyn****')
# print(first_GooSyn)


file_Answer = synonyms_data.drop(columns=['question', '0', '1', '2', '3'])
# print('\n')
# print('*****file_Answer****')
# print(file_Answer)
file_answerVec = file_Answer.to_numpy()
# print('\n')
# print('*****file_answerVec****')
# print(file_answerVec)


truth_vect = []
for i in range(len(syn_vec)):
    if(file_answerVec[i] == first_GooSyn[i]):
        truth_vect.append('correct')
    else:
        truth_vect.append('wrong')

# print('\n')
# print('*****truth_vect****')
# print(truth_vect)

final_Vector = [(synonyms_vec), (first_GooSyn), (truth_vect)]
# print('\n')
# print('*****final_Vector****')
# print(final_Vector)
csv_vec = list(zip(*final_Vector))


final_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
final_dict['Question'] = syn_vec.tolist()
final_dict['Answer'] = list(chain.from_iterable(file_answerVec))
final_dict['Model Guess'] = first_GooSyn
final_dict['Label'] = truth_vect
# print('\n')
# print('*****final_dict****')
# print(final_dict)

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


# ******************************************TASK 2******************************************
# ********************glove-twitter-200   &   glove-wiki-gigaword-200***********************
# ******************************************************************************************


glovTwitt200_model = api.load('glove-twitter-200')

GlTwitt200_result = []

# Glo50_result = {"Question Word": [],
#                 "Predicted": [], "Synonyme": [], "Result": []}


for i in (syn_vec):
    try:
        GlTwitt200_result.append(
            glovTwitt200_model.most_similar(positive=i, topn=1))
    except:
        GlTwitt200_result.append([('None Existing', 0)])


# print('*****Glo50_result****')
# print(Glo50_result)


flat_Twitt200Result = list(chain.from_iterable(GlTwitt200_result))

# print('\n')
# print('*****flat_Glov50Result****')
# print(flat_Glov50Result)
first_Twitt200 = []

for i in flat_Twitt200Result:
    first_Twitt200.append(i[0])


# print('\n')
# print('*****first_Gl50****')
# print(first_Gl50)


file_AnswerTwitt200 = synonyms_data.drop(
    columns=['question', '0', '1', '2', '3'])

# print('\n')
# print('*****file_AnswerGL50****')
# print(file_AnswerGL50)
file_Twitt200Vec = file_AnswerTwitt200.to_numpy()

# print('\n')
# print('****file_ansGL50Vec****')
# print(file_ansGL50Vec)


truth_vectTwitt200 = []

for i in range(len(syn_vec)):
    if(file_Twitt200Vec[i] == first_Twitt200[i]):
        truth_vectTwitt200.append('correct')
    else:
        truth_vectTwitt200.append('wrong')

# print('\n')
# print('*****truth_vectGL50****')
# print(truth_vectGL50)

final_VectorTwitt200 = [(synonyms_vec), (first_Twitt200), (file_Twitt200Vec)]

# print('\n')
# print('*****final_VectorTwitt200****')
# print(final_VectorTwitt200)
csv_vec = list(zip(*final_VectorTwitt200))


Twitt200final_dict = {}

fields = ['Question', 'Answer', 'Model Guess', 'Label']
Twitt200final_dict['Question'] = syn_vec.tolist()
Twitt200final_dict['Answer'] = list(chain.from_iterable(file_Twitt200Vec))
Twitt200final_dict['Model Guess'] = first_Twitt200
Twitt200final_dict['Label'] = truth_vectTwitt200
# print('\n')
# print('*****final_dict****')
# print(Twitt200final_dict)


(pd.DataFrame.from_dict(data=Twitt200final_dict, orient='columns')).to_csv(
    'glove-twitter-200.csv', header=True)

cT200 = truth_vectTwitt200.count('correct')
vT200 = len(first_Twitt200) - first_Twitt200.count('None Existing')
accuT200 = cT200/vT200


Twitt200_analysList = ['glove-twitter-200',
                       '1193514', str(cT200), str(vT200), str(accuT200)]


# ******************************************************************************************

wiki200_model = api.load("glove-wiki-gigaword-200")
wiki200_result = []

# Glo50_result = {"Question Word": [],
#                 "Predicted": [], "Synonyme": [], "Result": []}


for i in (syn_vec):
    try:
        wiki200_result.append(wiki200_model.most_similar(positive=i, topn=1))
    except:
        wiki200_result.append([('None Existing', 0)])


# print('*****Glo50_result****')
# print(Glo50_result)


flat_wiki200Result = list(chain.from_iterable(wiki200_result))

# print('\n')
# print('*****flat_Glov50Result****')
# print(flat_Glov50Result)
first_wiki200 = []


for i in flat_wiki200Result:
    first_wiki200.append(i[0])


# print('\n')
# print('*****first_Gl50****')
# print(first_Gl50)


file_Answerwiki200 = synonyms_data.drop(
    columns=['question', '0', '1', '2', '3'])

# print('\n')
# print('*****file_AnswerGL50****')
# print(file_AnswerGL50)
file_Wiki200Vec = file_Answerwiki200.to_numpy()

# print('\n')
# print('****file_ansGL50Vec****')
# print(file_ansGL50Vec)


truth_vectWiki200 = []

for i in range(len(syn_vec)):
    if(file_Wiki200Vec[i] == first_wiki200[i]):
        truth_vectWiki200.append('correct')
    else:
        truth_vectWiki200.append('wrong')

# print('\n')
# print('*****truth_vectGL50****')
# print(truth_vectGL50)

final_VectorWiki200 = [(synonyms_vec), (first_wiki200), (file_Wiki200Vec)]

# print('\n')
# print('*****final_VectorWiki200****')
# print(final_VectorWiki200)
csv_vec = list(zip(*final_VectorWiki200))


wiki200final_dict = {}

fields = ['Question', 'Answer', 'Model Guess', 'Label']
wiki200final_dict['Question'] = syn_vec.tolist()
wiki200final_dict['Answer'] = list(chain.from_iterable(file_Wiki200Vec))
wiki200final_dict['Model Guess'] = first_wiki200
wiki200final_dict['Label'] = truth_vectWiki200
# print('\n')
# print('*****final_dict****')
# print(wiki200final_dict)


(pd.DataFrame.from_dict(data=wiki200final_dict, orient='columns')).to_csv(
    'glove-wiki-gigaword-200.csv', header=True)

cW200 = truth_vectWiki200.count('correct')
vW200 = len(first_wiki200) - first_wiki200.count('None Existing')
accuW200 = cW200/vW200


wiki200_analysList = ['glove-wiki-gigaword-200',
                      '400000', str(cW200), str(vW200), str(accuW200)]


# ******************************************TASK 3******************************************
# ********************glove-wiki-gigaword-50   &   glove-wiki-gigaword-100******************
# ******************************************************************************************


Glo50_result = []
glov_model50 = api.load("glove-wiki-gigaword-50")
# Glo50_result = {"Question Word": [],
#                 "Predicted": [], "Synonyme": [], "Result": []}


for i in (syn_vec):
    try:
        Glo50_result.append(glov_model50.most_similar(positive=i, topn=1))
    except:
        Glo50_result.append([('None Existing', 0)])


# print('*****Glo50_result****')
# print(Glo50_result)


flat_Glov50Result = list(chain.from_iterable(Glo50_result))
# print('\n')
# print('*****flat_Glov50Result****')
# print(flat_Glov50Result)
first_Gl50 = []

for i in flat_Glov50Result:
    first_Gl50.append(i[0])


# print('\n')
# print('*****first_Gl50****')
# print(first_Gl50)


file_AnswerGL50 = synonyms_data.drop(columns=['question', '0', '1', '2', '3'])
# print('\n')
# print('*****file_AnswerGL50****')
# print(file_AnswerGL50)
file_ansGL50Vec = file_AnswerGL50.to_numpy()
# print('\n')
# print('****file_ansGL50Vec****')
# print(file_ansGL50Vec)


truth_vectGL50 = []

for i in range(len(syn_vec)):
    if(file_ansGL50Vec[i] == first_Gl50[i]):
        truth_vectGL50.append('correct')
    else:
        truth_vectGL50.append('wrong')

# print('\n')
# print('*****truth_vectGL50****')
# print(truth_vectGL50)

final_VectorFL50 = [(synonyms_vec), (first_Gl50), (file_ansGL50Vec)]
# print('\n')
# print('*****final_VectorFL50****')
# print(final_VectorFL50)
csv_vec = list(zip(*final_VectorFL50))


GL50final_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
GL50final_dict['Question'] = syn_vec.tolist()
GL50final_dict['Answer'] = list(chain.from_iterable(file_ansGL50Vec))
GL50final_dict['Model Guess'] = first_Gl50
GL50final_dict['Label'] = truth_vectGL50
# print('\n')
# print('*****final_dict****')
# print(GL50final_dict)


(pd.DataFrame.from_dict(data=GL50final_dict, orient='columns')).to_csv(
    'glove-wiki-gigaword-50.csv', header=True)

c50 = truth_vectGL50.count('correct')
v50 = len(first_Gl50) - first_Gl50.count('None Existing')
accu50 = c50/v50


Gl50_analysList = ['glove-wiki-gigaword-50',
                   '400000', str(c50), str(v50), str(accu50)]


# ******************************************************************************************


glov_model100 = api.load("glove-wiki-gigaword-100")
Glo100_result = []

# Glo50_result = {"Question Word": [],
#                 "Predicted": [], "Synonyme": [], "Result": []}


for i in (syn_vec):
    try:
        Glo100_result.append(glov_model100.most_similar(positive=i, topn=1))
    except:
        Glo100_result.append([('None Existing', 0)])


# print('*****Glo50_result****')
# print(Glo50_result)


flat_Glov100Result = list(chain.from_iterable(Glo100_result))
# print('\n')
# print('*****flat_Glov50Result****')
# print(flat_Glov50Result)
first_Gl100 = []

for i in flat_Glov100Result:
    first_Gl100.append(i[0])


# print('\n')
# print('*****first_Gl50****')
# print(first_Gl50)


file_AnswerGL100 = synonyms_data.drop(columns=['question', '0', '1', '2', '3'])
# print('\n')
# print('*****file_AnswerGL50****')
# print(file_AnswerGL50)
file_ansGL100Vec = file_AnswerGL100.to_numpy()
# print('\n')
# print('****file_ansGL50Vec****')
# print(file_ansGL50Vec)


truth_vectGL100 = []

for i in range(len(syn_vec)):
    if(file_ansGL100Vec[i] == first_Gl100[i]):
        truth_vectGL100.append('correct')
    else:
        truth_vectGL100.append('wrong')

# print('\n')
# print('*****truth_vectGL50****')
# print(truth_vectGL50)

final_VectorFL100 = [(synonyms_vec), (first_Gl100), (file_ansGL100Vec)]
# print('\n')
# print('*****final_VectorFL100****')
# print(final_VectorFL100)
csv_vec = list(zip(*final_VectorFL100))


GL100final_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
GL100final_dict['Question'] = syn_vec.tolist()
GL100final_dict['Answer'] = list(chain.from_iterable(file_ansGL100Vec))
GL100final_dict['Model Guess'] = first_Gl100
GL100final_dict['Label'] = truth_vectGL100
# print('\n')
# print('*****final_dict****')
# print(GL100final_dict)


(pd.DataFrame.from_dict(data=GL100final_dict, orient='columns')).to_csv(
    'glove-wiki-gigaword-100.csv', header=True)

c100 = truth_vectGL100.count('correct')
v100 = len(first_Gl100) - first_Gl100.count('None Existing')
accu100 = c100/v100


Gl100_analysList = ['glove-wiki-gigaword-100',
                    '400000', str(c100), str(v100), str(accu100)]


# ******************************************************************************************


with open('analysis.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    # for key, value in analysis_dict.items():
    writer.writerow(Goog_analysList)
    writer.writerow(Twitt200_analysList)
    writer.writerow(wiki200_analysList)
    writer.writerow(Gl50_analysList)
    writer.writerow(Gl100_analysList)

csvfile.close()
