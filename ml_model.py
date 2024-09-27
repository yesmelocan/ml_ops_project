import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sqlite3 as sql
from sklearn.model_selection import train_test_split

my_query = "SELECT * FROM insurance_train"

with sql.connect("insurance.sqlite") as connection:
    df = pd.read_sql_query (my_query,connection)    
connection.close


num =['id', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response']
not_num = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']

data =pd.get_dummies(df, columns=not_num,drop_first=True)

data.rename(columns={
    'Vehicle_Age_< 1 Year': 'Vehicle_Age_greater 1 Year',
    'Vehicle_Age_> 2 Years': 'Vehicle_Age_lesser 2 Years',
    'Vehicle_Damage_Yes': 'Vehicle_Damage_Yes'
}, inplace=True)

X = data.drop(['Response','id'], axis=1)
y = data['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 , random_state=1)


RF_model=RandomForestClassifier(n_estimators=100,random_state=1)
RF_model.fit(X_train, y_train)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

print(f"{knn.score(x_test,y_test)}")
print(f"{xgb.score(x_test,y_test)}")

