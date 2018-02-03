from __future__ import division
import numpy as np
import pandas as pd
import pickle
import os
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# read data
df_train = pd.read_table("../data/train_ai-lab.txt",names=["id","s1","s2","score"],sep="\t")
df_test = pd.read_table("../data/test_ai-lab.txt",names=["id",'s1','s2'],sep="\t")

df = pd.concat([df_train, df_test], axis=0,ignore_index=True)
print(df.shape)
# Letter n-gram
if os.path.isfile('./fea_data/cv_char.pkl') and os.path.isfile('./fea_data/ch_freq.pkl'):
    with open('./fea_data/cv_char.pkl', 'rb') as f:
        cv_char = pickle.load(f)
    with open('./fea_data/ch_freq.pkl', 'rb') as f:
        ch_freq = pickle.load(f)
else:
    cv_char = CountVectorizer(ngram_range=(1, 3), analyzer='char')
    ch_freq = np.array(cv_char.fit_transform(df['s1'].tolist() + df['s1'].tolist()).sum(axis=0))[0,:]
    with open('./fea_data/cv_char.pkl', 'wb') as f:
        pickle.dump(cv_char, f)
    with open('./fea_data/ch_freq.pkl', 'wb') as f:
        pickle.dump(ch_freq, f)

# get unigrams, bigrams, trigrams
unigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 1])
print(unigrams.values())
ix_unigrams = np.sort(list(unigrams.values()))
print('Unigrams:', len(unigrams))
bigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 2])
ix_bigrams = np.sort(list(bigrams.values()))
print('Bigrams: ', len(bigrams))
trigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 3])
ix_trigrams = np.sort(list(trigrams.values()))
print('Trigrams:', len(trigrams))

# get m_q1 && m_q2
def save_sparse_csr(fname, sm):
    np.savez(fname, 
             data=sm.data, 
             indices=sm.indices,
             indptr=sm.indptr, 
             shape=sm.shape)

def load_sparse_csr(fname):
    loader = np.load(fname)
    return sparse.csr_matrix((
        loader['data'], 
        loader['indices'], 
        loader['indptr']),
        shape=loader['shape'])

if os.path.isfile('./fea_data/m_q1.npz') and os.path.isfile('./fea_data/m_q2.npz'):
    m_q1 = load_sparse_csr('./fea_data/m_q1.npz')
    m_q2 = load_sparse_csr('./fea_data/m_q2.npz')
else:
    m_q1 = cv_char.transform(df['s1'].values)
    m_q2 = cv_char.transform(df['s2'].values)
    save_sparse_csr('./fea_data/m_q1.npz', m_q1)
    save_sparse_csr('./fea_data/m_q2.npz', m_q2)
"""
this is all about unigram featutes
"""
# unigram_jaccard
v_num = (m_q1[:, ix_unigrams] > 0).minimum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
v_den = (m_q1[:, ix_unigrams] > 0).maximum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_unigram_jaccard'] = v_score

# unigram_all_jaccard
v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
v_den = m_q1[:, ix_unigrams].sum(axis=1) + m_q2[:, ix_unigrams].sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_unigram_all_jaccard'] = v_score

# unigram_all_jaccard_max
v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
v_den = m_q1[:, ix_unigrams].maximum(m_q2[:, ix_unigrams]).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_unigram_all_jaccard_max'] = v_score

"""
This is all about bigram features 
"""
# bigram_jaccard
v_num = (m_q1[:, ix_bigrams] > 0).minimum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
v_den = (m_q1[:, ix_bigrams] > 0).maximum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_bigram_jaccard'] = v_score

# bigram_all_jaccard
v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
v_den = m_q1[:, ix_bigrams].sum(axis=1) + m_q2[:, ix_bigrams].sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_bigram_all_jaccard'] = v_score

# bigram_all_jaccard_max
v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
v_den = m_q1[:, ix_bigrams].maximum(m_q2[:, ix_bigrams]).sum(axis=1)
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_bigram_all_jaccard_max'] = v_score


"""
this is all about trigrams features
"""
m_q1 = m_q1[:, ix_trigrams]
m_q2 = m_q2[:, ix_trigrams]

# trigram_jaccard
v_num = (m_q1 > 0).minimum((m_q2 > 0)).sum(axis=1)
v_den = (m_q1 > 0).maximum((m_q2 > 0)).sum(axis=1)
v_den[np.where(v_den == 0)] = 1
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_trigram_jaccard'] = v_score

# trigram_all_jaccard
v_num = m_q1.minimum(m_q2).sum(axis=1)
v_den = m_q1.sum(axis=1) + m_q2.sum(axis=1)
v_den[np.where(v_den == 0)] = 1
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_trigram_all_jaccard'] = v_score

# trigram_all_jaccard_max
v_num = m_q1.minimum(m_q2).sum(axis=1)
v_den = m_q1.maximum(m_q2).sum(axis=1)
v_den[np.where(v_den == 0)] = 1
v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]

df['f_trigram_all_jaccard_max'] = v_score

# trigram_tfidf_cosine
tft = TfidfTransformer(
    norm='l2', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_num = np.array(m_q1_tf.multiply(m_q2_tf).sum(axis=1))[:, 0]
v_den = np.array(np.sqrt(m_q1_tf.multiply(m_q1_tf).sum(axis=1)))[:, 0] * \
        np.array(np.sqrt(m_q2_tf.multiply(m_q2_tf).sum(axis=1)))[:, 0]
v_num[np.where(v_den == 0)] = 1
v_den[np.where(v_den == 0)] = 1

v_score = 1 - v_num/v_den

df['f_trigram_tfidf_cosine'] = v_score

# trigram_tfidf_l2_euclidean
tft = TfidfTransformer(
    norm='l2', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['f_trigram_tfidf_l2_euclidean'] = v_score

# trigram_tfidf_l1_euclidean 	
tft = TfidfTransformer(
    norm='l1', 
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)
v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['f_trigram_tfidf_l1_euclidean'] = v_score

# trigram_tf_l2_euclidean
tft = TfidfTransformer(
    norm='l2', 
    use_idf=False, 
    smooth_idf=True, 
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['f_trigram_tf_l2_euclidean'] = v_score


# svd = TruncatedSVD(n_components=100)
# m_svd = svd.fit_transform(sparse.csc_matrix(sparse.vstack((m_q1_tf, m_q2_tf))))
#
# with open('1_svd.pkl', 'wb') as f:
#         pickle.dump(svd, f)
# with open('1_m_svd.npz', 'wb') as f:
#         np.savez(f, m_svd)
# df['f_q1_q2_tf_svd0'] = m_svd[:, 0]
#df['f_q1_q2_tf_svd0'].to_csv('ms-m_q1_q2_tf_svd0.csv')

ix_train = df[~df["score"].isnull()]
ix_test = df[df["score"].isnull()]
print(ix_train.shape, ix_test.shape)
feature = [x for x in list(ix_train.columns) if x not in ["s1",'s2']]
ix_train[feature].to_csv('../feature/train_ix.csv',index = False)
feature.remove("score")
ix_test[feature].to_csv('../feature/test_ix.csv',index = False)