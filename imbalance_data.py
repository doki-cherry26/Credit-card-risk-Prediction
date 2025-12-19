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
logging=setup_logging('imbalance_data')
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
from ALL_MODELS import common


def balance_data(X_train,y_train,X_test,y_test):
    try:
        logging.info(f' Before Number of rows for Good class:{sum(y_train==1)}')
        logging.info(f' Before Number of rows for Bad class:{sum(y_train==0)}')
        sms_reg=SMOTE(random_state=42)
        X_train_bal,y_train_bal=sms_reg.fit_resample(X_train,y_train)
        logging.info(f' After Number of rows for Good class:{sum(y_train_bal == 1)}')
        logging.info(f' After Number of rows for Bad class:{sum(y_train_bal == 0)}')
        logging.info(f'{X_train_bal.shape}')
        logging.info(f'{y_train_bal.shape}')
        logging.info(f'{X_train_bal.sample(10)}')
        sc=StandardScaler()
        sc.fit(X_train_bal)
        X_train_bal_scaled=sc.transform(X_train_bal)
        X_test_scaled = sc.transform(X_test)
        logging.info(f'{X_train_bal_scaled}')
        with open('scalar.pkl','wb') as f:
            pickle.dump(sc,f)
        common (X_train_bal_scaled,y_train_bal,X_test_scaled,y_test)


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logging.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')



