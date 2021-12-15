from logging import exception
from os import remove
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
import re


Dataset = pd.read_csv('synonyms.csv', delimiter=',')
syn_Choices = Dataset.drop(columns=['question'])
list_Syn_Choices = [list(row) for row in syn_Choices.values]
# synonyms_QA = pd.DataFrame(synonyms_data, columns=['question','answer'])
print('\n')
print('*****list_Syn_Choices****')
print(list_Syn_Choices)
print(len(list_Syn_Choices))

syn_Ques = Dataset.drop(columns=['answer', '0', '1', '2', '3'])
list_Syn_Ques = [list(row) for row in syn_Ques.values]
list_Syn_Ques = list(chain.from_iterable(list_Syn_Ques))
print('\n')
print('*****list_Syn_Ques****')
print(list_Syn_Ques)
print(len(list_Syn_Ques))


syn_Ans = Dataset.drop(columns=['question', '0', '1', '2', '3'])
list_Syn_Ans = [list(row) for row in syn_Ans.values]
list_Syn_Ans = list(chain.from_iterable(list_Syn_Ans))
print('\n')
print('*****list_Syn_Ans****')
print(list_Syn_Ans)
print(len(list_Syn_Ans))


google_news5Guess = []
google_news5Guess = []
# google_news5Guess = []
google_news5Guess = []
Glove_Twitt5Guess = []


def getSyn_Model(modelGuess_list, list_Syn_Ques, model_name):
    modelGuess = []
    for i in (list_Syn_Ques):
        try:
            modelGuess.append(model_name.most_similar(positive=i, topn=5))
        except:
            modelGuess.append([('None Existing', 0)])
    modelGuess = list(chain.from_iterable(modelGuess))
    for i in modelGuess:
        modelGuess_list.append(i[0])
    # n = 5
    # modelGuess_list = [modelGuess_list[i:i+n]
    #                    for i in range(0, len(modelGuess_list), n)]
    return modelGuess_list


# ******************************************************************************************
Google_model = api.load("word2vec-google-news-300")
print('\n')
print(Google_model.most_similar(positive='principal', topn=5))

# Lists to be used with getSyn_Model --> (modelGuess_list) returns 2d list of 5 guesses for each question


getSyn_Model(google_news5Guess, list_Syn_Ques, Google_model)

n = 5
google_news5Guess = [google_news5Guess[i:i+n]
                     for i in range(0, len(google_news5Guess), n)]


# google_news5Guess[49] = 'None'


print('\n')
print('*****google_news5Guess****')
print(google_news5Guess)
print(len(google_news5Guess))

common_Google = []
truth_vectGoogle = []


for i in range(len(google_news5Guess)):
    if(google_news5Guess[i] == 'None Existing'):
        common_Google.extend(list_Syn_Choices[i][0])
        truth_vectGoogle.append('guess')
    elif (set(google_news5Guess[i]).intersection(list_Syn_Choices[i])):
        common_Google.extend(
            set(google_news5Guess[i]).intersection(list_Syn_Choices[i]))
        truth_vectGoogle.append('correct')
    else:
        common_Google.append(google_news5Guess[i][0])
        truth_vectGoogle.append('wrong')

# del common_Google[49]

print('\n')
print('*****common_Google****')
print(common_Google)
print(len(common_Google))
print('\n')
print('*****truth_vectGoogle****')
print(truth_vectGoogle)
print(len(truth_vectGoogle))

GoogleFinal_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
GoogleFinal_dict['Question'] = list_Syn_Ques
GoogleFinal_dict['Answer'] = list_Syn_Ans
GoogleFinal_dict['Model Guess'] = common_Google
GoogleFinal_dict['Label'] = truth_vectGoogle

(pd.DataFrame.from_dict(data=GoogleFinal_dict, orient='columns')).to_csv(
    'word2vec-google-news-300.csv', header=True)

c_google = truth_vectGoogle.count('correct')
v_google = c_google + truth_vectGoogle.count('wrong')
accu_google = c_google/v_google

Google_analysList = ['word2vec-google-news-300',
                     '400000', str(c_google), str(v_google), str(accu_google)]


with open('analysis.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    # for key, value in analysis_dict.items():
    writer.writerow(Google_analysList)
