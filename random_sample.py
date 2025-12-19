import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from logging_code import setup_logging
logging=setup_logging('random_sample')


def random_sample_imputation_technique(X_train,X_test):
    try:
        logging.info(f'Total rows in training data : {X_train.shape}')
        logging.info(f'Total rows in testing data : {X_test.shape}')
        logging.info(f"Before technique X_train : {X_train.columns}")
        logging.info(f"Before technique X_test : {X_test.columns}")
        logging.info(f"Before technique X_train : {X_train.isnull().sum()}")
        logging.info(f"Before technique X_test : {X_test.isnull().sum()}")

        for i in X_train.columns:
            if X_train[i].isnull().sum() > 0:
                logging.info(f"Train Column name : {i}")
                X_train[i+"_replaced"] = X_train[i].copy()
                X_test[i+"_replaced"] = X_test[i].copy()
                s1 = X_train[i].dropna().sample(X_train[i].isnull().sum(),random_state=42)
                s2 = X_test[i].dropna().sample(X_test[i].isnull().sum(), random_state=42)
                s1.index = X_train[X_train[i].isnull()].index
                s2.index = X_test[X_test[i].isnull()].index
                X_train.loc[X_train[i].isnull() , i+"_replaced"] = s1
                X_test.loc[X_test[i].isnull(), i+"_replaced"] = s2
                X_train = X_train.drop([i],axis=1)
                X_test = X_test.drop([i],axis=1)

        logging.info(f"After technique X_train : {X_train.columns}")
        logging.info(f"After technique X_test : {X_test.columns}")
        logging.info(f"After technique X_train : {X_train.isnull().sum()}")
        logging.info(f"After technique X_test : {X_test.isnull().sum()}")
        logging.info(f'Total rows in training data : {X_train.shape}')
        logging.info(f'Total rows in testing data : {X_test.shape}')

        return X_train,X_test

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logging.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')