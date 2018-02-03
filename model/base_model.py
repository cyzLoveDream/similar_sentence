# 传统机器学习所使用的模型
class model:
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def svr(self, c=5, g = 3):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler
        import scipy as sc
        #  索引最好的c = 5 gamma =1
        # bag of words c = le3 gamma = 0.1
        # tfidf c=2.8, g=2
        s = make_pipeline(RobustScaler(),SVR(kernel='rbf', C=10,gamma=0.005))
        s.fit(self.X_train,self.y_train)
        print("the model is SVR and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, s.predict(self.X_test))[0])
        return s
    def rf(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestRegressor
        import scipy as sc
        # count = 0
        # while count < 20:
        #     import random
        #     i = random.randint(0,1000)
            # 590
        rf = make_pipeline(RandomForestRegressor(random_state=641,n_estimators =250,max_depth=9))
        rf.fit(self.X_train,self.y_train)
        print("the model is rf and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, rf.predict(self.X_test))[0])
            # count += 1
        return rf
    def gboost(self):
        from sklearn.ensemble import GradientBoostingRegressor
        import scipy as sc

        GBoost = GradientBoostingRegressor(n_estimators=330, learning_rate=0.01,
                                           max_depth=12, max_features='sqrt',
                                           min_samples_leaf=1, min_samples_split=42,
                                           loss='ls', random_state =40,subsample=1)
        GBoost.fit(self.X_train,self.y_train)
        print("the model is gboost and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, GBoost.predict(self.X_test))[0])
        return GBoost
    def xgboost(self):
        from xgboost import XGBRegressor
        import xgboost as xgb
        import scipy as sc
        # [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500]
        # while count < 20:
        #     import random
        #     i = random.randint(0,10000)
        #     # 1024
        model_xgb = xgb.XGBRegressor(colsample_bytree=1, gamma=5,
                                     learning_rate=0.01, max_depth=11,
                                     min_child_weight=1.7817, n_estimators=500,
                                     reg_alpha=0.8, reg_lambda=5,
                                     subsample=0.5213, silent=1,
                                     seed =1024, nthread = -1)
        model_xgb.fit(self.X_train,self.y_train)
        print("the model is xgboost and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, model_xgb.predict(self.X_test))[0])
            # count += 1
        return model_xgb
    def lgb(self):
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        import scipy as sc
        model_lgb = LGBMRegressor(objective='regression',num_leaves=4,
                                  learning_rate=0.05, n_estimators=290,
                                  max_bin = 147, subsample = 0.65,
                                  colsample_bytree = 0.7,
                                  feature_fraction_seed=46, subsample_freq=9,
                                  min_child_samples =20, min_child_weight=0.001)
        model_lgb.fit(self.X_train,self.y_train)
        print("the model is lgb and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, model_lgb.predict(self.X_test))[0])
        return model_lgb
    def stacking(self):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler,MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
        from xgboost import XGBRegressor
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        import xgboost as xgb
        from mlxtend.regressor import StackingRegressor
        import scipy as sc
        s = make_pipeline(RobustScaler(),SVR(kernel='rbf', C=10,gamma=0.005))
        rf = make_pipeline(RandomForestRegressor(random_state=641,n_estimators =250,max_depth=9))
        GBoost = GradientBoostingRegressor(n_estimators=330, learning_rate=0.01,
                                           max_depth=12, max_features='sqrt',
                                           min_samples_leaf=1, min_samples_split=42,
                                           loss='ls', random_state =40,subsample=1)
        model_xgb = xgb.XGBRegressor(colsample_bytree=1, gamma=5,
                                     learning_rate=0.01, max_depth=11,
                                     min_child_weight=1.7817, n_estimators=500,
                                     reg_alpha=0.8, reg_lambda=5,
                                     subsample=0.5213, silent=1,
                                     seed =1024, nthread = -1)
        model_lgb = LGBMRegressor(objective='regression',num_leaves=4,
                                  learning_rate=0.05, n_estimators=290,
                                  max_bin = 147, subsample = 0.65,
                                  colsample_bytree = 0.7,
                                  feature_fraction_seed=46, subsample_freq=9,
                                  min_child_samples =20, min_child_weight=0.001)
        regressors = [s,rf,GBoost, model_lgb,model_xgb]
        stregr = StackingRegressor(regressors=regressors, meta_regressor=model_xgb)
        stregr.fit(self.X_train,self.y_train)
        print("the model is staking and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, stregr.predict(self.X_test))[0])
        return stregr
    class _AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

        def __init__(self, models):
            self.models = models
        # we define clones of the original models to fit the data in
        def fit(self, X, y):
            from sklearn.base import clone
            self.models_ = [clone(x) for x in self.models]
            # Train cloned base models
            for model in self.models_:
                model.fit(X, y)
            return self
        #Now we do the predictions for cloned models and average them
        def predict(self, X):
            predictions = np.column_stack([
                model.predict(X) for model in self.models_
            ])
            return np.mean(predictions, axis=1)
    def average(self):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler,MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
        from xgboost import XGBRegressor
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        import xgboost as xgb
        from mlxtend.regressor import StackingRegressor
        from sklearn.kernel_ridge import KernelRidge
        import scipy as sc
        #         self._load_package()
        # c=7,g=0.075
        s = make_pipeline(RobustScaler(),SVR(kernel='rbf', C=10,gamma=0.005))
        rf = make_pipeline(RandomForestRegressor(random_state=641,n_estimators =250,max_depth=9))
        GBoost = GradientBoostingRegressor(n_estimators=330, learning_rate=0.01,
                                           max_depth=12, max_features='sqrt',
                                           min_samples_leaf=1, min_samples_split=42,
                                           loss='ls', random_state =40,subsample=1)
        model_xgb = xgb.XGBRegressor(colsample_bytree=1, gamma=5,
                                     learning_rate=0.01, max_depth=11,
                                     min_child_weight=1.7817, n_estimators=500,
                                     reg_alpha=0.8, reg_lambda=5,
                                     subsample=0.5213, silent=1,
                                     seed =1024, nthread = -1)
        model_lgb = LGBMRegressor(objective='regression',num_leaves=4,
                                  learning_rate=0.05, n_estimators=290,
                                  max_bin = 147, subsample = 0.65,
                                  colsample_bytree = 0.7,
                                  feature_fraction_seed=46, subsample_freq=9,
                                  min_child_samples =20, min_child_weight=0.001)
        regressors = [rf,GBoost, model_lgb,model_xgb,s]
        stregr = StackingRegressor(regressors=regressors, meta_regressor=model_xgb)
        averaged_models =self._AveragingModels(models = (rf,stregr,model_xgb,s))
        averaged_models.fit(self.X_train,self.y_train)
        print("the model is average and the test's pearsonr is: ", sc.stats.pearsonr(self.y_test, averaged_models.predict(self.X_test))[0])
        return averaged_models

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer
def load_data():
    trains_fea = pd.read_csv("../feature/train_doc2vec4.csv")
    trains_fea1 = pd.read_csv("../feature/train_gram_feature.csv")
    trains_fea1.drop("id",axis=1, inplace=True)
    trains_fea1.drop("score",axis=1,inplace=True)
    trains_fea2 = pd.read_csv("../feature/train_ix.csv")
    trains_fea2.drop("id",axis=1, inplace=True)
    trains_fea2.drop("score",axis=1,inplace=True)
    trains_fea3 = pd.read_csv("../feature/train_simple_feature.csv")
    trains_fea3.drop("id",axis=1, inplace=True)
    trains_fea3.drop("score",axis=1,inplace=True)
    trains_fea4 = pd.read_csv("../feature/train_weight_noweight.csv")
    trains_fea4.drop("id",axis=1, inplace=True)
    trains_fea4.drop("score",axis=1,inplace=True)
    trains_fea5 = pd.read_csv("../feature/train_weight_tfidf.csv")
    trains_fea5.drop("id",axis=1, inplace=True)
    trains_fea5.drop("score",axis=1,inplace=True)
    trains = pd.concat([trains_fea, trains_fea1, trains_fea2, trains_fea3,trains_fea4,trains_fea5],axis=1)
    test_fea = pd.read_csv("../feature/test_doc2vec4.csv")
    test_fea1 = pd.read_csv("../feature/test_gram_feature.csv")
    test_fea1.drop("id",axis=1, inplace=True)
    test_fea2 = pd.read_csv("../feature/test_ix.csv")
    test_fea2.drop("id",axis=1, inplace=True)
    test_fea3 = pd.read_csv("../feature/test_simple_feature.csv")
    test_fea3.drop("id",axis=1, inplace=True)
    test_fea4 = pd.read_csv("../feature/test_weight_noweight.csv")
    test_fea4.drop("id",axis=1, inplace=True)
    test_fea5 = pd.read_csv("../feature/test_weight_tfidf.csv")
    test_fea5.drop("id",axis=1, inplace=True)
    tests = pd.concat([test_fea,test_fea1,test_fea2,test_fea3,test_fea4,test_fea5],axis=1)
    return trains, tests
trains, tests = load_data()
trains.fillna(value=0,inplace=True)
tests.fillna(value = 0,inplace=True)
print(trains.shape, tests.shape)
# print(list(trains.columns), len(list(trains.columns)))
# data = pd.concat([trains, tests],axis=0)
# bow_data = data[["id","s1","s2","score"]]
# bow_data["words"] = bow_data["s1"] + bow_data["s2"]
# bow_test = bow_data[bow_data.score.isnull()]
# bow_train = bow_data[~bow_data.score.isnull()]
# list_test = bow_test["words"].tolist()
# list_corpus = bow_train["words"].tolist()
# list_labels = bow_train["score"].tolist()
# list_corpus
lables = trains["score"].values
feature = [x for x in list(trains.columns) if x not in ["id","score"]]
for f in feature:
    trains[f] = trains[f].apply(lambda x: round(x, 5))
    tests[f] = tests[f].apply(lambda x:round(x,5))
# print(feature, len(feature))
X_train, X_test, y_train, y_test = train_test_split(trains[feature], lables, test_size=0.2, random_state=42)
# print(X_train.shape, X_test.shape)
# print(list(X_train.columns))
# m = model(np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test))
# m.svr(c=7,g=0.075)
m1 = model(np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test))
m1.svr()
m1.rf()
m1.xgboost()
m1.lgb()
m1.gboost()
average = m1.average()
stacking = m1.stacking()
# test_pd["score"] = average.predict(test_counts.toarray())
# test_pd["score"] = test_pd["score"].apply(lambda x: 0 if x < 0 else (5 if x >5 else x))
# test_pd["score"].describe()
# test_pd[["id","score"]].to_csv("submission_sample",index=False, header=False)
# !pwd && ls && head -n 5 submission_sample
# !wget -nv -O kesci_submit https://cdn.kesci.com/submit_tool/v1/kesci_submit&&chmod +x kesci_submit
# !./kesci_submit -token 你的token号 -file submission_sample