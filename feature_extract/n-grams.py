#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import datetime
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
# nltk.download()
# plt.show()
stops = set(stopwords.words("english"))
from nltk import ngrams
from simhash import Simhash
from multiprocessing import Pool

def tokenize(sequence):
    #words = word_tokenize(sequence)
    words = sequence.split(' ')
    #filtered_words = [word for word in words if word not in stopwords.words('english')]
    return words

def readData():
    trains = pd.read_table("../data/train_ai-lab.txt",names=["id","s1","s2","score"],sep="\t")
    tests = pd.read_table("../data/test_ai-lab.txt",names=["id",'s1','s2'],sep="\t")
    return trains, tests


def Jaccarc(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = len(q1words) + len(q2words) + 1
    same = 0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2):
                same += 1
    return float(same) / (float(tot - same)+0.000001)


def Dice(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = len(q1words) + len(q2words) + 1
    same = 0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2):
                same += 1
    return 2 * float(same) / (float(tot)+0.000001)

def Ochiai(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = len(q1words) * len(q2words) + 1
    same = 0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2):
                same += 1
    return float(same) / (np.sqrt(float(tot))+0.000001)

#n-grams 
def get_word_ngrams(sequence, n=3):
    tokens = tokenize(sequence)
    return [' '.join(ngram) for ngram in ngrams(tokens, n)]


def get_character_ngrams(sequence, n=3):
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]

def caluclate_simhash_distance(sequence1, sequence2):
    return Simhash(sequence1).distance(Simhash(sequence2))


def get_word_distance(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return caluclate_simhash_distance(q1, q2)

def get_word_2gram_distance(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_char_2gram_distance(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)

def get_word_3gram_distance(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)

def get_char_3gram_distance(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)



def get_word_distance2(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return Jaccarc(q1, q2)

def get_word_2gram_distance2(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return Jaccarc(q1, q2)

def get_char_2gram_distance2(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return Jaccarc(q1, q2)

def get_word_3gram_distance2(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return Jaccarc(q1, q2)

def get_char_3gram_distance2(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return Jaccarc(q1, q2)




def get_word_distance3(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return Dice(q1, q2)

def get_word_2gram_distance3(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return Dice(q1, q2)

def get_char_2gram_distance3(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return Dice(q1, q2)

def get_word_3gram_distance3(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return Dice(q1, q2)

def get_char_3gram_distance3(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return Dice(q1, q2)




def get_word_distance4(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return Ochiai(q1, q2)

def get_word_2gram_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return Ochiai(q1, q2)

def get_char_2gram_distance4(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return Ochiai(q1, q2)

def get_word_3gram_distance4(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return Ochiai(q1, q2)

def get_char_3gram_distance4(sentence):
    q1, q2 = sentence.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return Ochiai(q1, q2)

def makeFeature(df_features):
    pool = Pool(processes=20)
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('get n-grams')
    df_features['f_1dis'] = pool.map(get_word_distance, df_features['sentence'])
    df_features['f_2word_dis'] = pool.map(get_word_2gram_distance, df_features['sentence'])
    df_features['f_2char_dis'] = pool.map(get_char_2gram_distance, df_features['sentence'])
    df_features['f_3word_dis'] = pool.map(get_word_3gram_distance, df_features['sentence'])
    df_features['f_3char_dis'] = pool.map(get_char_3gram_distance, df_features['sentence'])


    df_features['f_1dis2'] = pool.map(get_word_distance2, df_features['sentence'])
    df_features['f_2word_dis2'] = pool.map(get_word_2gram_distance2, df_features['sentence'])
    df_features['f_2char_dis2'] = pool.map(get_char_2gram_distance2, df_features['sentence'])
    df_features['f_3word_dis2'] = pool.map(get_word_3gram_distance2, df_features['sentence'])
    df_features['f_3char_dis2'] = pool.map(get_char_3gram_distance2, df_features['sentence'])



    df_features['f_1dis3'] = pool.map(get_word_distance3, df_features['sentence'])
    df_features['f_2word_dis3'] = pool.map(get_word_2gram_distance3, df_features['sentence'])
    df_features['f_2char_dis3'] = pool.map(get_char_2gram_distance3, df_features['sentence'])
    df_features['f_3word_dis3'] = pool.map(get_word_3gram_distance3, df_features['sentence'])
    df_features['f_3char_dis3'] = pool.map(get_char_3gram_distance3, df_features['sentence'])



    df_features['f_1dis4'] = pool.map(get_word_distance4, df_features['sentence'])
    df_features['f_2word_dis4'] = pool.map(get_word_2gram_distance4, df_features['sentence'])
    df_features['f_2char_dis4'] = pool.map(get_char_2gram_distance4, df_features['sentence'])
    df_features['f_3word_dis4'] = pool.map(get_word_3gram_distance4, df_features['sentence'])
    df_features['f_3char_dis4'] = pool.map(get_char_3gram_distance4, df_features['sentence'])
    print('all done')
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    df_features.fillna(0.0)
    return df_features

if __name__ == "__main__":
    train, test = readData()
    train['sentence'] = train['s1'] + '_split_tag_' + train['s2']
    test['sentence'] = test['s1'] + '_split_tag_' + test['s2']

    train = makeFeature(train)
    feature = [x for x in list(train.columns) if x not in ["s1",'s2','sentence']]
    train[feature].to_csv('../feature/train_gram_feature.csv', index=False)

    test = makeFeature(test)
    feature.remove("score")
    test[feature].to_csv('../feature/test_gram_feature.csv', index=False)


