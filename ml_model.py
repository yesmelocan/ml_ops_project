import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sqlite3 as sql
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
from sklearn.naive_bayes import GaussianNB

my_query = "SELECT * FROM insurance_train"

with sql.connect("insurance.sqlite") as connection:
    df = pd.read_sql_query (my_query,connection)    
connection.close


num =['id', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response']
not_num = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']



X = df.drop(['Response','id','Gender', 'Vehicle_Age', 'Vehicle_Damage'], axis=1)
y = df['Response']
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 , random_state=1)


NB_model = GaussianNB()
NB_model.fit(X_train, y_train)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

print(f"{NB_model.score(X_test,y_test)}")
print(f"{xgb.score(X_test,y_test)}")

pickle.dump(NB_model,open("NB_model.pkl","wb"))
pickle.dump(xgb,open("qgb_model.pkl","wb"))