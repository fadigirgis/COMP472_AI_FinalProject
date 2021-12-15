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


google_news5Guess = []
Glove50_5guess = []
Glove100_5guess = []
Glovew200_5guess = []
GloveTw200_5guess = []


# glovTw_model200 = api.load("glove-twitter-200")
# print('\n')
# print(glovTw_model200.most_similar(positive='principal', topn=5))

# Lists to be used with getSyn_Model --> (modelGuess_list) returns 2d list of 5 guesses for each question


# getSyn_Model(GloveTw200_5guess, list_Syn_Ques, glovTw_model200)

GloveTw200_5guess = [['immensely', 'tremendously', 'hugely', 'displeased', 'massively'], ['provision', 'amendments', 'regulations', 'legislation', 'considerations'], ['discounting', 'irruptive', 'co-option', 'inflooding', 'correspondingly'], ['influential', 'high-profile', 'activists', 'journalist', 'activist'], ['lausanne', 'zenith', 'montpellier', 'strasbourg', 'toulouse'], ['imperfect', 'inherently', 'fundamentally', 'premise', 'reasoning'], ['urgent', 'required', 'needed', 'desperately', 'neede'], ['consuming', 'consume', 'ingested', 'consumption', 'eaten'], ['calender', 'calendar', 'xmas', 'christmas', 'easter'], ['succinctly', 'cosias', 'crucially', 'extensively', 'accouterments'], ['saluting', 'praises', 'applauds', 'bows', 'salute'], ['confinement', 'secluded', 'sentenced', 'prisoners', 'desolate'], ['minimise', 'nasrallah', 'stabilise', 'avenge', 'conclude'], ['determination', 'persistence', 'hardwork', 'humility', 'dedication'], ['fatuous', 'vacuous', 'insipid', 'short-sighted', 'aspirational'], ['took', 'showing', 'looked', 'came', 'knew'], ['continuously', 'wanting', 'complain', 'always', 'trying'], ['issue', 'problems', 'concerns', 'problem', 'policy'], ['furnishing', 'renovate', 'refurbish', 'allocate', 'dispose'], ['pricey', 'expensive', 'disastrous', 'avoid', 'ineffective'], ['recognised', 'honored', 'respected', 'established', 'recognize'], ['place', 'right', 'there', 'corner', 'spots'], ['making', 'makes', 'made', 'can', 'it'], ['usually', 'sometimes', 'rarely', 'however', 'frequently'], ['tactless', 'argumentative', 'aloof', 'personable', 'well-liked'], ['debates', 'obama', 'romney', 'election', 'presidential'], ['minded', 'clear', 'broad', 'shallow', 'path'], ['arrange', 'arrangement', 'discussed', 'organised', 'finalized'], ['shinee', 'exo', 'bigbang', 'b.a.p', 'sunggyu'], ['ostentatious', 'confrontational', 'temperament', 'revengeful', 'arogant'], ['sanctioned', 'lauded', 'incurred', 'faulted', 'heaped'], ['deconstruct', 'tryi', 'navigate', 'transparently', 'devising'], ['distributing', 'distributed', 'contribute', 'evenly', 'disseminate'], ['inconsistencies', 'irregularities', 'discrepancy', 'inaccuracies', 'caveats'], ['influential', 'eloquent', 'marketable', 'impactful', 'acclaimed'], ['unparalleled', 'professionalism', 'durability', 'utmost', 'unsurpassed'], ['None Existing', 'None Existing', 'None Existing', 'None Existing', 'None Existing'], ['hua', 'hum', 'hain', 'tou', 'nai'], ['vande', 'jai', 'bharat', 'matram', 'salman'],
                     ['highlights', 'definitely', 'performance', 'feature', 'week'], ['hurriedly', 'thoughtfully', 'clumsily', 'reluctantly', 'unwillingly'], ['fabulously', 'womanly', 'fionn', 'unaffected', 'non-political'], ['smirk', 'grins', 'grinning', 'smiles', 'smirks'], ['abusing', 'abused', 'assaulted', 'physically', 'harassed'], ['practitioner', 'physicians', 'nurse', 'assistant', 'pediatric'], ['basically', 'practically', 'ultimately', 'opposed', 'largely'], ['interested', 'pleased', 'impressed', 'excellent', 'eager'], ['somwhere', 'located', 'somewere', 'residing', 'neatly'], ['mayor', 'general', 'director', 'local', 'central'], ['quickly', 'falling', 'rapidly', 'moving', 'turning'], ['build', 'building', 'designed', 'developed', 'created'], ['task', 'chores', 'to-do', 'simplest', 'assignments'], ['likely', 'perhaps', 'however', 'outcome', 'although'], ['None Existing', 'None Existing', 'None Existing', 'None Existing', 'None Existing'], ['chronology', 'resultant', 'co-option', 'overtop', 'auspices'], ['crazily', 'weirdly', 'terribly', 'absurdly', 'hugely'], ['lauded', 'hails', 'touted', 'praised', 'heralded'], ['force', 'forces', 'guard', 'authority', 'military'], ['penned', 'authored', 'devised', 'orchestrated', 'negated'], ['clients', 'colleagues', 'applicants', 'employers', 'recruits'], ['usually', 'often', 'genuinely', 'rarely', 'tend'], ['injuries', 'sustaining', 'suffered', 'caused', 'continued'], ['hellish', 'efficacious', 'arduous', 'precipice', 'skirmish'], ['tranquility', 'uniformity', 'lightness', 'partisanship', 'decadence'], ['disintegrate', 'deflate', 'wither', 'fanciful', 'dissappear'], ['solely', 'mainly', 'largely', 'specifically', 'purely'], ['vernacular', 'pejorative', 'derogatory', 'nomenclature', 'utilitarian'], ['solved', 'addressed', 'rectified', 'resolves', 'resolving'], ['doable', 'plausible', 'palatable', 'economically', 'achievable'], ['lbvvvs', 'premises', 'fastly', 'introspect', 'ngalahke'], ['percentages', 'highest', 'percent', 'lowest', 'ratio'], ['terminating', 'revoked', 'voided', 're-opened', 'reopened'], ['uniforms', 'wear', 'wearing', 'shirt', 'jacket'], ['figures', 'guess', 'trying', 'find', 'how'], ['adequate', 'require', 'insufficient', 'obtain', 'provide'], ['style', 'clothing', 'designer', 'beauty', 'dress'], ['advertised', 'touted', 'depicted', 'indoctrinated', 'predominantly'], ['smaller', 'larger', 'than', 'huge', 'small'], ['reggae', 'rock', 'country', 'growing', 'hip'], ['usually', 'although', 'however', 'hardly', 'able']]


# rowLeng = []
# for row in GloveTw200_5guess:
#     rowLeng.append(len(row))

# max_length = max(rowLeng)


# for row in GloveTw200_5guess:
#     while len(row) < max_length:
#         row.append(('None Existing', 0))

# n = 5
# GloveTw200_5guess = [GloveTw200_5guess[i:i+n]
#                      for i in range(0, len(GloveTw200_5guess), n)]


# GloveTw200_5guess[49] = 'None'


print('\n')
print('*****GloveTw200_5guess****')
print(GloveTw200_5guess)
print(len(GloveTw200_5guess))

commonGLTw200 = []
truth_vectGLTw200 = []


# for i in range(len(GloveTw200_5guess)):
#     if(GloveTw200_5guess[i][0] == 'None Existing'):
#         commonGLTw200.append(list_Syn_Choices[i])
#         truth_vectGLTw200.append('guess')
#     elif (set(GloveTw200_5guess[i]).intersection(set(list_Syn_Choices[i]))):
#         set_guess = (set(GloveTw200_5guess[i]).intersection(
#             set(list_Syn_Choices[i])))
#         guess_list = list(set_guess)
#         if ((len(guess_list) == 0) or (guess_list[0] == 'None Existing')):
#             commonGLTw200.append(list_Syn_Choices[i])
#             truth_vectGLTw200.append('guess')
#         commonGLTw200.append(guess_list[0])
#         truth_vectGLTw200.append('correct')
#     else:
#         commonGLTw200.append(GloveTw200_5guess[i][0])
#         truth_vectGLTw200.append('wrong')


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
