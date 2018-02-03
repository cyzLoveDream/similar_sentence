#Import Initial Packages
import numpy as np
import pandas as pd
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import datetime


tokenizer = RegexpTokenizer(r'\w+')
stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def Cosine(vec1, vec2):
    vec1 = np.array(vec1,dtype=np.float)
    vec2 = np.array(vec2,dtype=np.float)
    Lx = np.sqrt(vec1.dot(vec1))
    Ly = np.sqrt(vec2.dot(vec2))
    return vec1.dot(vec2) / ((Lx * Ly)+0.000001)

def Manhatton(vec1, vec2):
    return np.sum(np.fabs(np.array(vec1,dtype=np.float) - np.array(vec2,dtype=np.float)))

def Euclidean(vec1, vec2):
    return np.sqrt(np.sum(np.array(vec1,dtype=np.float) - np.array(vec2,dtype=np.float)) ** 2)

def PearsonSimilar(vec1, vec2):
    vec1 = np.array(vec1,dtype=np.float)
    vec2 = np.array(vec2,dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('pearson')[0][1]

def SpearmanSimilar(vec1, vec2):
    vec1 = np.array(vec1,dtype=np.float)
    vec2 = np.array(vec2,dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('spearman')[0][1]

def KendallSimilar(vec1, vec2):
    vec1 = np.array(vec1,dtype=np.float)
    vec2 = np.array(vec2,dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('kendall')[0][1]

def get_processed_text(text=""):
    """
    Remove stopword,lemmatizing the words and remove special character to get important content
    """
    clean_text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)
    tokens = tokenizer.tokenize(clean_text)
    tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens
              if token not in stopwords and len(token) >= 2]
    return tokens


model_name = "%s" %("doc_2_vec_model4")
model_saved_file = "./doc2vec_model/%s" %(model_name)
model = gensim.models.doc2vec.Doc2Vec.load(model_saved_file)

def get_question_vector(question1 = ""):
    question_token1 = get_processed_text(question1)
    tokenize_text1 = ' '.join(question_token1)
    tokenize_text1 = gensim.utils.simple_preprocess(tokenize_text1)
    infer_vector_of_question1 = model.infer_vector(tokenize_text1)
    return infer_vector_of_question1




def makeFeature(df_features):
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print ('get sentence vector')
    df_features['doc2vec1'] = df_features.s1.map(lambda x: get_question_vector(x))
    df_features['doc2vec2'] = df_features.s2.map(lambda x: get_question_vector(x))
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print ('get six kinds of coefficient about vector')
    df_features['z3_cosine'] = df_features.apply(lambda x: Cosine(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_manhatton'] = df_features.apply(lambda x: Manhatton(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_euclidean'] = df_features.apply(lambda x: Euclidean(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_pearson'] = df_features.apply(lambda x: PearsonSimilar(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_spearman'] = df_features.apply(lambda x: SpearmanSimilar(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_kendall'] = df_features.apply(lambda x: KendallSimilar(x['doc2vec1'], x['doc2vec2']), axis=1)
    del df_features["doc2vec1"]
    del df_features["doc2vec2"]
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    return df_features

if __name__ == "__main__":
    train = pd.read_table("../data/train_ai-lab.txt",names=["id","s1","s2","score"],sep="\t")
    test = pd.read_table("../data/test_ai-lab.txt",names=["id",'s1','s2'],sep="\t")
    train = makeFeature(train)
    feature = [x for x in list(train.columns) if x not in ["s1",'s2','sentence']]
    train[feature].to_csv('../feature/train_doc2vec4.csv', index=False)

    test = makeFeature(test)
    feature.remove("score")
    test[feature].to_csv('../feature/test_doc2vec4.csv', index=False)
    print(train.shape, test.shape)
