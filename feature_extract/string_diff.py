#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
import datetime
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))


def readData():
    trains = pd.read_table("../data/train_ai-lab.txt",names=["id","s1","s2","score"],sep="\t")
    tests = pd.read_table("../data/test_ai-lab.txt",names=["id",'s1','s2'],sep="\t")
    return trains, tests


def total_unique_words(row):
    return len(set(row['s1']).union(row['s2']))

def total_unq_words_stop(row, stops =stops):
    return len([x for x in set(row['s1']).union(row['s2']) if x not in stops])

def wc_diff(row):
    return abs(len(row['s1']) - len(row['s2']))

def wc_ratio(row):
    l1 = len(row['s1'])*1.0
    l2 = len(row['s2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(row['s1'])) - len(set(row['s2'])))

def wc_ratio_unique(row):
    l1 = len(set(row['s1'])) * 1.0
    l2 = len(set(row['s2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=stops):
    return abs(len([x for x in set(row['s1']) if x not in stops]) - len([x for x in set(row['s2']) if x not in stops]))

def wc_ratio_unique_stop(row, stops=stops):
    l1 = len([x for x in set(row['s1']) if x not in stops])*1.0
    l2 = len([x for x in set(row['s2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['s1'] or not row['s2']:
        return np.nan
    return int(row['s1'][0] == row['s2'][0])

def char_diff(row):
    return abs(len(''.join(row['s1'])) - len(''.join(row['s2'])))

def char_ratio(row):
    l1 = len(''.join(row['s1']))
    l2 = len(''.join(row['s2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=stops):
    return abs(len(''.join([x for x in set(row['s1']) if x not in stops])) - len(''.join([x for x in set(row['s2']) if x not in stops])))

def makeFeature(df_features):
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    df_features['f_total_unique_words'] = df_features.apply(total_unique_words, axis=1, raw=True)
    df_features['f_total_unq_words_stop'] = df_features.apply(total_unq_words_stop, axis=1, raw=True)
    df_features['f_wc_diff'] = df_features.apply(wc_diff, axis=1, raw=True) 
    df_features['f_wc_ratio'] = df_features.apply(wc_ratio,axis=1, raw=True)
    df_features['f_wc_diff_unique']= df_features.apply(wc_diff_unique, axis=1, raw=True)
    df_features['f_wc_ratio_unique']= df_features.apply(wc_ratio_unique, axis=1, raw=True)
    df_features['f_wc_diff_unique_stop']= df_features.apply(wc_diff_unique_stop, axis=1, raw=True)
    df_features['f_wc_ratio_unique_stop']= df_features.apply(wc_ratio_unique_stop, axis=1, raw=True)
    df_features['f_same_start_word']= df_features.apply(same_start_word, axis=1, raw=True)
    df_features['f_char_diff']= df_features.apply(char_diff, axis=1, raw=True)
    df_features['f_char_ratio']= df_features.apply(char_ratio, axis=1, raw=True)
    df_features['f_char_diff_unique_stop']= df_features.apply(char_diff_unique_stop, axis=1, raw=True)
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    return df_features




if __name__ == "__main__":
    train, test = readData()

    train = makeFeature(train)
    feature = [x for x in list(train.columns) if x not in ["s1",'s2','sentence']]
    train[feature].to_csv('../feature/train_simple_feature.csv', index=False)

    test = makeFeature(test)
    feature.remove("score")
    test[feature].to_csv('../feature/test_simple_feature.csv', index=False)
