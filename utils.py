import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 


def genClf(clf, param_dist, X_train, X_test, y_train, name, ovr=False):
    
    print("-"*50)
    print(f"Generating '{name}' MultiClass classifier...")
    
    # if we want to, wrap the classifier in the ovr scikit-learn wrapper
        # it builds a classifier for each label
    if ovr:
        clf = OneVsRestClassifier(clf(**param_dist)).fit(X_train, y_train)
    else:
        clf = clf(**param_dist).fit(X_train, y_train)        

    scoring = {'acc': 'accuracy', 'roc_auc_ovr': 'roc_auc_ovr', 'roc_auc_ovr_weights': 'roc_auc_ovr_weighted'}
    cv_data = cross_validate(clf, X_train, y_train, scoring=scoring, cv=3, n_jobs=-1)
    keys = ['acc', 'roc_auc_ovr', 'roc_auc_ovr_weights']
    
    scores = [np.mean(cv_data["test_"+key]) for key in keys]
    ret_scores = {name : scores}
    
    return clf, ret_scores



def model_search(X_train, X_test, y_train, testing_data):
    """ Tests an assortment of models to heuristically determine the best choice
    
    Returns
    -------
    (list(object), list(list(float)))
        a tuple of all the trained classifier objects and a dataframe of scores
    """
    # initialize return objects
    all_scores = {}
    all_clfs = []
    
    # iterate through testing data
    for i in testing_data:
        # generate the clf and score
        clf, ret_scores = genClf(i['clf'], i['param_dist'], X_train, X_test, y_train, name=i['name'], ovr=i['ovr'])
        # append the return values
        all_clfs.append(clf)
        all_scores.update(ret_scores)
    
    # craft dataframe of all scores
    df_scores = pd.DataFrame.from_dict(all_scores, 
                                       orient="index", 
                                       columns=["raw_acc", "macro_roc_auc_ovr", "weighted_roc_auc_ovr"])

    return all_clfs, df_scores


# convert data to standard form (scaled by standard deviation around the mean)
def standardize(feature):
    mu = feature.mean() # compute mean
    stdev = feature.std() # compute standard deviation
    return (feature - mu) / stdev

def preprocessor(df, test=False):
    # encode addresses and labels for ease of access
    le = preprocessing.LabelEncoder()
#     df['address'] = le.fit_transform(df.address)
    if not test:
        df['label'] = le.fit_transform(df.label)
    # sort values by date and reset index
    df['date'] = df.year + np.round(df.day / 365,3)
    df = df.sort_values(by='date').reset_index(drop=True)
    # add address counts, number of times given address appears in the data
    df['address_count'] = df.groupby('address')['address'].transform('size')
    # add cumulative count column, cumulative number of times address appears in the data
    df['address_cumcount'] = df.groupby('address').cumcount() # add cumuative address counts to dataframe
    
    #standardize the data
    to_standardize = ['length','weight','count','looped','neighbors','income']
    standardized_features = df[to_standardize].apply(standardize)
    df = df.drop(to_standardize,axis=1).join(df[to_standardize].apply(standardize))
    return df, le



class MasterModel:
    """ Master model object for performing three steps in ransomware classification
    
    Parameters
    ----------
    dictKnownAddress : dict()
        the dictionary of known ransomware adresses
    binaryClf : object
        the binary classifier for whether or not a transaction is ransomware. Needs to be already fitted!
    multiclassClf : object
        the multiclass classifier for, if it is a ransomware transaction, which family is it in. Needs to be already fitted!
        
    """
    def __init__(self, dictKnownAddress, binaryClf, multiclassClf, is_fit=False):
        self.dictKnownAddress = dictKnownAddress
        self.binaryClf = binaryClf
        self.multiclassClf = multiclassClf
    
    
    def predictions(self, X):
        ret = self.predict(X)
        
        return ret
            
    
    def predict(self, X):
        
        df_preds = X.address.map(self.dictKnownAddress)
        
        # for all those not in the dictKnownAddress, replace with 0
        # for all 0's binaryClf classify
        df_preds[df_preds.isna()] = self.binaryClf.predict(X[df_preds.isna()]) 
        df_preds.replace(1, 28)
        
        # if it is still 0, it is ransomware, so familial classify
        df_preds[df_preds == 0] = self.multiclassClf.predict(X[df_preds == 0])
        
        return df_preds