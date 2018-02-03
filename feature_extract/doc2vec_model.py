#Import Initial Packages
import pandas as pd
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re 
from collections import namedtuple
import multiprocessing
import datetime
import warnings
warnings.filterwarnings("ignore")
# import nltk
# import matplotlib.pyplot as plt
# nltk.download()
# plt.show()
tokenizer = RegexpTokenizer(r'\w+')
stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

df_train = pd.read_table("../data/train_ai-lab.txt",names=["id","s1","s2","score"],sep="\t")
df_test = pd.read_table("../data/test_ai-lab.txt",names=["id",'s1','s2'],sep="\t")
df_train_set1 = df_train[["id","s1"]]
df_train_set2 = df_train[["id","s2"]]
df_train_set1.columns = ["id","s"]
df_train_set2.columns =["id","s"]
df_test_set1 = df_train[["id","s1"]]
df_test_set2 = df_train[["id","s2"]]
df_test_set1.columns = ["id","s"]
df_test_set2.columns =["id","s"]
df_train_set = pd.concat([df_train_set1,df_train_set2,df_test_set1,df_test_set2],axis=0)

#Language Processing
def get_processed_text(text=""):
    """
    Remove stopword,lemmatizing the words and remove special character to get important content
    """
    clean_text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)
    tokens = tokenizer.tokenize(clean_text)
    tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens
              if token not in stopwords and len(token) >= 2]
    return tokens

#Process and clean up traing set
alldocuments = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')       
keywords = []
for index,record in df_train_set[:].iterrows():
    qid = str(record["id"])
    sentence = str(record["s"])
    tokens = get_processed_text(sentence)
    words = tokens
    words_text = " ".join(words)
    words = gensim.utils.simple_preprocess(words_text)
    tags = [qid]
    alldocuments.append(analyzedDocument(words, tags))


def train_and_save_doc2vec_model(alldocuments,document_model="model4",m_iter=100,m_min_count=2,m_size=100,m_window=5):
    print ("Start Time : %s" %(str(datetime.datetime.now())))
    #Train Model
    cores = multiprocessing.cpu_count()
    saved_model_name = "./doc2vec_model/doc_2_vec_%s" %(document_model)
    doc_vec_file = "%s" %(saved_model_name)
    if document_model == "model1":
        # PV-DBOW 
        model_1 = gensim.models.Doc2Vec(alldocuments,dm=0,workers=cores,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter,dbow_words=1)
        model_1.save("%s" %(doc_vec_file))
        print ("model training completed : %s" %(doc_vec_file))
    elif document_model == "model2":
        # PV-DBOW 
        model_2 = gensim.models.Doc2Vec(alldocuments,dm=0,workers=cores,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter,dbow_words=0)
        model_2.save("%s" %(doc_vec_file))
        print ("model training completed : %s" %(doc_vec_file))
    elif document_model == "model3":
        # PV-DM w/average
        model_3 = gensim.models.Doc2Vec(alldocuments,dm=1, dm_mean=1,size=m_size, window=m_window,min_count=m_min_count,iter=m_iter)
        model_3.save("%s" %(doc_vec_file))
        print("model training completed : %s" %(doc_vec_file))

    elif document_model == "model4":
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        model_4 = gensim.models.Doc2Vec(alldocuments,dm=1, dm_concat=1,workers=cores, size=m_size, window=m_window,min_count=m_min_count,iter=m_iter)
        model_4.save("%s" %(doc_vec_file))
        print ("model training completed : %s" %(doc_vec_file))
    print ("Record count %s" %len(alldocuments))
    print ("End Time %s" %(str(datetime.datetime.now())))

train_and_save_doc2vec_model(alldocuments)


