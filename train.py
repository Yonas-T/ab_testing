from dbm import dumb
import os 
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import dvc.api
from joblib import dump

import mlflow
import mlflow.sklearn


from fast_ml.model_development import train_valid_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold
from sklearn.linear_model import LogisticRegression

from sklearn import tree, metrics

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class Train:

    def __init__(self):
        self

    def read_data():
        df_split = pd.read_csv('data/data_with_response.csv')
        return df_split

    def label_encoder(df = read_data()):
        label_encoder = LabelEncoder()

        df['experiment'] = label_encoder.fit_transform(df['experiment'])
        df['date'] = label_encoder.fit_transform(df['date'])
        df['hour'] = label_encoder.fit_transform(df['hour'])
        df['device_make'] = label_encoder.fit_transform(df['device_make'])
        df['platform_os'] = label_encoder.fit_transform(df['platform_os'])
        df['browser'] = label_encoder.fit_transform(df['browser'])
        df['user_response'] = label_encoder.fit_transform(df['user_response'])
        return df

    def browser_data_split_os(self, df = label_encoder()):
        df_Platform_copied = df.copy()
        df_platform = df_Platform_copied.drop(columns='platform_os', axis=1)
        df_platform.to_csv('data/data_with_os.csv', index=False)
        return df_platform

    def browser_data_split_browser(self, df = label_encoder()):
        df_browser_copied = df.copy()
        df_browser = df_browser_copied.drop(columns='browser', axis=1)
        df_browser.to_csv('data/data_with_platform.csv', index=False)
        
        return df_browser

    def access_data_from_dvc(self, path, df = label_encoder()):
        # self.browser_data_split_os()
        # self.browser_data_split_browser()
        
        df_browser_copied = df.copy()
        df_browser = df_browser_copied.drop(columns='browser', axis=1)
        df_browser.to_csv('data/data_with_platform.csv', index=False)
        
        # with dvc.api.open(
        # path,
        # mode='rb',
        # ) as data:
        #      df_from_dvc = pd.read_csv(data)
        #      df_from_dvc.head()
        #      df_from_dvc.drop(['auction_id'],axis=1,inplace=True)
        

        data_url= dvc.api.get_url(
                "data/browser_clean_data.csv",
        repo="https://github.com/Yonas-T/ab_testing/tree/environments_setup",)

        mlflow.log_param("data_url", data_url)

        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])

        df_from_dvc = pd.read_csv(data_url)
        print(df_from_dvc.head())
        return df_from_dvc

    def data_split(self, path):
        
        data_df = self.access_data_from_dvc(path)
        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(data_df, target = 'user_response', 
                                                                            train_size=0.7, valid_size=0.2, test_size=0.1)
        print(X_train.shape), print(y_train.shape)
        print(X_valid.shape), print(y_valid.shape)
        print(X_test.shape), print(y_test.shape)
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def train_model(self):
        MODEL_PATH = os.path.join(os.getcwd(), "models")
        MODEL_PATH_LRM = os.path.join(MODEL_PATH, "clf_lrm.joblib")

        mlflow.set_experiment("SmartAD")


        X_train, y_train, _, _, X_test, _ = self.data_split('data/data_with_os.csv')
        model = LogisticRegression()
        model.fit(X_train, y_train)
        model_predictions = model.predict(X_test)
        result=cross_val_score(estimator=model,X=X_train,y=y_train,cv=5,scoring='accuracy')
        
        dump(model, 'MODEL_PATH_LRM')
        print(result)
        return result, model_predictions

if __name__ == "__main__":
    train = Train()
    warnings.filterwarnings('ignore')
   
    train.train_model()

    

