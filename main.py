"""
In this file we are going to load the data and other ML pipeline techniques
which are needed
"""
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
logging=setup_logging('main')
from random_sample import random_sample_imputation_technique
from var_out import variable_transformation_outliers
from feature_selection import complete_feature_selection
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from imbalance_data import  balance_data



class CREDIT_CARD_DATA:
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path)
            logging.info(f'total number of rows:{self.df.shape[0]}')
            logging.info(f'total number of columns:{self.df.shape[1]}')
            logging.info(f'Before:{self.df.isnull().sum()}')
            self.df = self.df.drop([150000, 150001], axis=0)
            self.df=self.df.drop(['MonthlyIncome.1'],axis=1)
            logging.info(f'---------------------------------------')
            logging.info(f'After:{self.df.isnull().sum()}')
            logging.info(f'------------------------------------------')
            for i in self.df.columns:
                if self.df[i].isnull().sum()>0:
                    logging.info(f'{i}->{self.df[i].dtype}')
                    if self.df[i].dtype=='object':
                        self.df[i]=pd.to_numeric(self.df[i])
                        logging.info(f'{i}->{self.df[i].dtype}')
                    else:
                        pass
            self.X=self.df.iloc[:,:-1]
            self.y=self.df.iloc[:,-1]
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logging.info(f'{self.X_train.columns}')
            logging.info(f'{self.X_test.columns}')

            logging.info(f'{self.y_train.sample(5)}')
            logging.info(f'{self.y_test.sample(5)}')

            logging.info(f'Training data size:{self.X_train.shape}')
            logging.info(f'Testing data size:{self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logging.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    def missing_value(self):
        try:
            logging.info(f'Total rows in training data : {self.X_train.shape}')
            logging.info(f'Total rows in testing data : {self.X_test.shape}')
            logging.info(f"Before : {self.X_train.columns}")
            logging.info(f"Before : {self.X_test.columns}")
            logging.info(f"Before : {self.X_train.isnull().sum()}")
            logging.info(f"Before : {self.X_test.isnull().sum()}")
            self.X_train, self.X_test = random_sample_imputation_technique(self.X_train, self.X_test)
            logging.info(f"After : {self.X_train.columns}")
            logging.info(f"After : {self.X_test.columns}")
            logging.info(f"After : {self.X_train.isnull().sum()}")
            logging.info(f"After : {self.X_test.isnull().sum()}")
            logging.info(f'Total rows in training data : {self.X_train.shape}')
            logging.info(f'Total rows in testing data : {self.X_test.shape}')


        except Exception as e:
             error_type, error_msg, error_line = sys.exc_info()
             logging.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def vt_hol(self):
        try:
            logging.info(f'{self.X_train.columns}')
            logging.info(f'{self.X_test.columns}')
            logging.info(f'---------------------------')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logging.info(f'--------------------------------------------')
            logging.info(f'{self.X_train_num.columns}')
            logging.info(f'{self.X_train_cat.columns}')
            logging.info(f'{self.X_test_num.columns}')
            logging.info(f'{self.X_test_cat.columns}')
            logging.info(f'{self.X_train_num.shape}')
            logging.info(f'{self.X_train_cat.shape}')
            logging.info(f'{self.X_test_num.shape}')
            logging.info(f'{self.X_test_cat.shape}')
            self.X_train_num,self.X_test_num=variable_transformation_outliers(self.X_train_num,self.X_test_num)
            logging.info(f'{self.X_train_num.columns}->{self.X_train_num.shape}')
            logging.info(f'{self.X_test_num.columns}->{self.X_test_num.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logging.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    def fs(self):
        try:
            logging.info(f' Before:{self.X_train_num.columns}->{self.X_train_num.shape}')
            logging.info(f' Before:{self.X_test_num.columns}->{self.X_test_num.shape}')
            self.X_train_num,self.X_test_num= complete_feature_selection(self.X_train_num,self.X_test_num,self.y_train)
            logging.info(f' After:{self.X_train_num.columns}->{self.X_train_num.shape}')
            logging.info(f' After:{self.X_test_num.columns}->{self.X_test_num.shape}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logging.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def cat_to_num(self):
        try:
            logging.info(f'{self.X_train_cat.columns}')
            logging.info(f'{self.X_test_cat.columns}')

            for i in self.X_train_cat.columns:
                logging.info(f'{i}->:{self.X_train_cat[i].unique()}')


            logging.info(f' Before converting:{self.X_train_cat}')
            logging.info(f' Before converting:{self.X_test_cat}')

            one_hot = OneHotEncoder(drop='first')
            one_hot.fit(self.X_train_cat[['Gender','Region']])
            result=one_hot.transform(self.X_train_cat[['Gender','Region']]).toarray()
            f=pd.DataFrame(data=result,columns=one_hot.get_feature_names_out())
            self.X_train_cat.reset_index(drop=True,inplace=True)
            f.reset_index(drop=True,inplace=True)
            self.X_train_cat=pd.concat([self.X_train_cat,f],axis=1)
            self.X_train_cat =self.X_train_cat.drop(['Gender','Region'],axis=1)

            result1 = one_hot.transform(self.X_test_cat[['Gender', 'Region']]).toarray()
            f1 = pd.DataFrame(data=result1, columns=one_hot.get_feature_names_out())
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f.reset_index(drop=True, inplace=True)
            self.X_test_cat = pd.concat([self.X_test_cat, f1], axis=1)
            self.X_test_cat = self.X_test_cat.drop(['Gender', 'Region'], axis=1)


            ord_end =OrdinalEncoder()
            ord_end.fit(self.X_train_cat[['Rented_OwnHouse','Occupation','Education']])
            result2= ord_end.transform(self.X_train_cat[['Rented_OwnHouse','Occupation','Education']])
            t=pd.DataFrame(data=result2,columns=ord_end.get_feature_names_out()+'_res')
            self.X_train_cat.reset_index(drop=True, inplace=True)
            t.reset_index(drop=True, inplace=True)
            self.X_train_cat=pd.concat([self.X_train_cat,t],axis=1)
            self.X_train_cat = self.X_train_cat.drop(['Rented_OwnHouse','Occupation','Education'], axis=1)

            result3 = ord_end.transform(self.X_test_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
            t1 = pd.DataFrame(data=result3, columns=ord_end.get_feature_names_out() + '_res')
            self.X_test_cat.reset_index(drop=True, inplace=True)
            t1.reset_index(drop=True, inplace=True)
            self.X_test_cat = pd.concat([self.X_test_cat, t1], axis=1)
            self.X_test_cat = self.X_test_cat.drop(['Rented_OwnHouse', 'Occupation', 'Education'], axis=1)

            logging.info(f'{self.X_train_cat.columns}')
            logging.info(f'{self.X_test_cat.columns}')

            logging.info(f' After converting:{self.X_train_cat}')
            logging.info(f' After converting:{self.X_test_cat}')

            logging.info(f'{self.X_train_cat.shape}')
            logging.info(f'{self.X_test_cat.shape}')

            logging.info(f'{self.X_train_cat.isnull().sum()}')
            logging.info(f'{self.X_test_cat.isnull().sum()}')


            self.X_train_num.reset_index(drop=True,inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data=pd.concat([self.X_train_num,self.X_train_cat],axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logging.info(f'{self.training_data.shape}')
            logging.info(f'{self.testing_data.shape}')

            logging.info(f'{self.training_data.isnull().sum()}')
            logging.info(f'{self.testing_data.isnull().sum()}')

            logging.info(f'=============================================================')

            logging.info(f'{self.training_data.sample(10)}')
            logging.info(f'{self.testing_data.sample(10)}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logging.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def data_b(self):
        try:
            self.y_train=self.y_train.map({'Good':1,'Bad':0}).astype(int)
            self.y_test = self.y_test.map({'Good': 1, 'Bad': 0}).astype(int)
            balance_data(self.training_data,self.y_train,self.testing_data,self.y_test)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logging.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')


if __name__=='__main__':
    try:
        obj=CREDIT_CARD_DATA(f'C:\\Users\\chari\\Downloads\\machine laearning pipeline project\\creditcard.csv')
        obj.missing_value()
        obj.vt_hol()
        obj.fs()
        obj.cat_to_num()
        obj.data_b()


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logging.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')



