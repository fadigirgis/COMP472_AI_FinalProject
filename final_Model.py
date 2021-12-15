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


# ******************************************************************************************
# ********************************** Synonyms CSV*******************************************
# ******************************************************************************************

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

NoneList = [('None Existing', 0) for i in range(5)]
# A method to return a list of 5 guesses from trained model


def getSyn_Model(modelGuess_list, list_Syn_Ques, model_name):
    modelGuess = []
    for i in (list_Syn_Ques):
        try:
            guess = []
            guess = model_name.most_similar(positive=i, topn=5)
            if (len(guess) != 5):
                guess.append(('None Existing', 0))
            modelGuess.append(guess)
        except:
            modelGuess.append(NoneList)
    print(modelGuess)
    modelGuess = list(chain.from_iterable(modelGuess))
    for i in modelGuess:
        modelGuess_list.append(i[0])
    # n = 5
    # modelGuess_list = [modelGuess_list[i:i+n]
    #                    for i in range(0, len(modelGuess_list), n)]
    return modelGuess_list


def getLists_intersection(GuessList, CommonList, TruthList):
    for i in range(len(GuessList)):
        if(GuessList[i][0] == 'None Existing'):
            CommonList.append(list_Syn_Choices[i])
            TruthList.append('guess')
        elif (set(GuessList[i]).intersection(set(list_Syn_Choices[i]))):
            set_guess = (set(GuessList[i]).intersection(
                set(list_Syn_Choices[i])))
            guess_list = list(set_guess)
            if ((len(guess_list) == 0) or (guess_list[0] == 'None Existing')):
                CommonList.append(list_Syn_Choices[i])
                TruthList.append('guess')
            CommonList.append(guess_list[0])
            TruthList.append('correct')
        else:
            CommonList.append(GuessList[i][0])
            TruthList.append('wrong')


google_news5Guess = []
Glove50_5guess = []
Glove100_5guess = []
Glovew200_5guess = []
GloveTw200_5guess = []


# ************************************** TASK 1 ********************************************
# ****************************** Google_model TASK 1 ***************************************
# ******************************************************************************************

Google_model = api.load("word2vec-google-news-300")
# print('\n')
# print(Google_model.most_similar(positive='principal', topn=5))

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

getLists_intersection(google_news5Guess, common_Google, truth_vectGoogle)

# for i in range(len(google_news5Guess)):
#     if(google_news5Guess[i] == 'None Existing'):
#         common_Google.extend(list_Syn_Choices[i][0])
#         truth_vectGoogle.append('guess')
#     elif (set(google_news5Guess[i]).intersection(list_Syn_Choices[i])):
#         common_Google.extend(
#             set(google_news5Guess[i]).intersection(list_Syn_Choices[i]))
#         truth_vectGoogle.append('correct')
#     else:
#         common_Google.append(google_news5Guess[i][0])
#         truth_vectGoogle.append('wrong')

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


# # ************************************** TASK 2 ********************************************
# # ********************glove-twitter-200   &   glove-wiki-gigaword-200***********************
# # ******************************************************************************************


# # ********************************** glove-wiki-gigaword-200 **************************************

glov_modelw200 = api.load("glove-wiki-gigaword-200")
# print('\n')
# print(glov_modelw200.most_similar(positive='principal', topn=5))

# Lists to be used with getSyn_Model --> (modelGuess_list) returns 2d list of 5 guesses for each question


getSyn_Model(Glovew200_5guess, list_Syn_Ques, glov_modelw200)

n = 5
Glovew200_5guess = [Glovew200_5guess[i:i+n]
                    for i in range(0, len(Glovew200_5guess), n)]


# Glove200_5guess[49] = 'None'


print('\n')
print('*****Glovew200_5guess****')
print(Glovew200_5guess)
print(len(Glovew200_5guess))

commonGLw200 = []
truth_vectGLw200 = []


getLists_intersection(Glovew200_5guess, commonGLw200, truth_vectGLw200)

# del commonGL200[49]

print('\n')
print('*****commonGLw200****')
print(commonGLw200)
print(len(commonGLw200))
print('\n')
print('*****truth_vectGLw200****')
print(truth_vectGLw200)
print(len(truth_vectGLw200))

GLw200final_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
GLw200final_dict['Question'] = list_Syn_Ques
GLw200final_dict['Answer'] = list_Syn_Ans
GLw200final_dict['Model Guess'] = commonGLw200
GLw200final_dict['Label'] = truth_vectGLw200

(pd.DataFrame.from_dict(data=GLw200final_dict, orient='columns')).to_csv(
    'glove-wiki-gigaword-200.csv', header=True)

cw200 = truth_vectGLw200.count('correct')
vw200 = cw200 + truth_vectGLw200.count('wrong')
accuw200 = cw200/vw200

Glw200_analysList = ['glove-wiki-gigaword-200',
                     '400000', str(cw200), str(vw200), str(accuw200)]


# # ********************************** glove-twitter-200 **************************************


glovTw_model200 = api.load("glove-twitter-200")
# print('\n')
# print(glovTw_model200.most_similar(positive='principal', topn=5))

# Lists to be used with getSyn_Model --> (modelGuess_list) returns 2d list of 5 guesses for each question


getSyn_Model(GloveTw200_5guess, list_Syn_Ques, glovTw_model200)


n = 5
GloveTw200_5guess = [GloveTw200_5guess[i:i+n]
                     for i in range(0, len(GloveTw200_5guess), n)]


# GloveTw200_5guess[49] = 'None'


print('\n')
print('*****GloveTw200_5guess****')
print(GloveTw200_5guess)
print(len(GloveTw200_5guess))

commonGLTw200 = []
truth_vectGLTw200 = []


getLists_intersection(GloveTw200_5guess, commonGLTw200, truth_vectGLTw200)


# del commonGLTw200[49]

print('\n')
print('*****commonGLTw200****')
print(commonGLTw200)
print(len(commonGLTw200))
print('\n')
print('*****truth_vectGLTw200****')
print(truth_vectGLTw200)
print(len(truth_vectGLTw200))

GLTw200final_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
GLTw200final_dict['Question'] = list_Syn_Ques
GLTw200final_dict['Answer'] = list_Syn_Ans
GLTw200final_dict['Model Guess'] = commonGLTw200
GLTw200final_dict['Label'] = truth_vectGLTw200

(pd.DataFrame.from_dict(data=GLTw200final_dict, orient='columns')).to_csv(
    'glove-twitter-200.csv', header=True)

c_t200 = truth_vectGLTw200.count('correct')
v_t200 = c_t200 + truth_vectGLTw200.count('wrong')
accu_t200 = c_t200/v_t200

GlTw200_analysList = ['glove-twitter-200',
                      '400000', str(c_t200), str(v_t200), str(accu_t200)]


# ****************************************** TASK 3 ******************************************
# ******************** glove-wiki-gigaword-50   &   glove-wiki-gigaword-100 ******************
# ********************************************************************************************


# ********************************** glove-wiki-gigaword-50 **************************************

glov_model50 = api.load("glove-wiki-gigaword-50")
print('\n')

# there was a bug in this model that returns a symbol for the word principal therefore
# had to delete that symbol from guesses later
# print(glov_model50.most_similar(positive='principal', topn=5))


# Lists to be used with getSyn_Model --> (modelGuess_list) returns 2d list of 5 guesses for each question


getSyn_Model(Glove50_5guess, list_Syn_Ques, glov_model50)

n = 5
Glove50_5guess = [Glove50_5guess[i:i+n]
                  for i in range(0, len(Glove50_5guess), n)]


# Glove50_5guess[49] = 'None'


print('\n')
print('*****Glove50_5guess****')
print(Glove50_5guess)
print(len(Glove50_5guess))

commonGL50 = []
truth_vectGL50 = []


getLists_intersection(Glove50_5guess, commonGL50, truth_vectGL50)

# del commonGL50[49]

print('\n')
print('*****commonGL50****')
print(commonGL50)
print(len(commonGL50))
print('\n')
print('*****truth_vectGL50****')
print(truth_vectGL50)
print(len(truth_vectGL50))

GL50final_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
GL50final_dict['Question'] = list_Syn_Ques
GL50final_dict['Answer'] = list_Syn_Ans
GL50final_dict['Model Guess'] = commonGL50
GL50final_dict['Label'] = truth_vectGL50

(pd.DataFrame.from_dict(data=GL50final_dict, orient='columns')).to_csv(
    'glove-wiki-gigaword-50.csv', header=True)

c50 = truth_vectGL50.count('correct')
v50 = c50 + truth_vectGL50.count('wrong')
accu50 = c50/v50

Gl50_analysList = ['glove-wiki-gigaword-50',
                   '400000', str(c50), str(v50), str(accu50)]


# ********************************** glove-wiki-gigaword-100 **************************************


glov_model100 = api.load("glove-wiki-gigaword-100")
# print('\n')
# print(glov_model100.most_similar(positive='principal', topn=5))

# Lists to be used with getSyn_Model --> (modelGuess_list) returns 2d list of 5 guesses for each question


getSyn_Model(Glove100_5guess, list_Syn_Ques, glov_model100)

n = 5
Glove100_5guess = [Glove100_5guess[i:i+n]
                   for i in range(0, len(Glove100_5guess), n)]


Glove100_5guess[49] = 'None'


print('\n')
print('*****Glove100_5guess****')
print(Glove100_5guess)
print(len(Glove100_5guess))

commonGL100 = []
truth_vectGL100 = []


getLists_intersection(Glove100_5guess, commonGL100, truth_vectGL100)

# del commonGL100[49]

print('\n')
print('*****commonGL100****')
print(commonGL100)
print(len(commonGL100))
print('\n')
print('*****truth_vectGL100****')
print(truth_vectGL100)
print(len(truth_vectGL100))

GL100final_dict = {}
fields = ['Question', 'Answer', 'Model Guess', 'Label']
GL100final_dict['Question'] = list_Syn_Ques
GL100final_dict['Answer'] = list_Syn_Ans
GL100final_dict['Model Guess'] = commonGL100
GL100final_dict['Label'] = truth_vectGL100

(pd.DataFrame.from_dict(data=GL100final_dict, orient='columns')).to_csv(
    'glove-wiki-gigaword-100.csv', header=True)

c100 = truth_vectGL100.count('correct')
v100 = c100 + truth_vectGL100.count('wrong')
accu100 = c100/v100

Gl100_analysList = ['glove-wiki-gigaword-100',
                    '400000', str(c100), str(v100), str(accu100)]


# ******************************************* Writting to Analysis File *******************************************


with open('analysis.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    # for key, value in analysis_dict.items():
    writer.writerow(Google_analysList)
    writer.writerow(GlTw200_analysList)
    writer.writerow(Glw200_analysList)
    writer.writerow(Gl50_analysList)
    writer.writerow(Gl100_analysList)

csvfile.close()
