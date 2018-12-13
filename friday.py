# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import dill
dill.load_session('friday.pkl')

import os
import numpy as np
import pandas as pd
from statistics import mode
os.chdir('C:\\Users\\Chandu\\Desktop\\IMAR DATA\\BlackFriday')
friday_train= pd.read_csv('train.csv')
friday_test= pd.read_csv('test.csv')

#-------------------------------------------------summary---------------
summary= friday_train.describe()
friday_train.columns.values
friday_train.dtypes
# ----------------continous-----'User_ID',  'Purchase','Occupation','Marital_Status',
 #      'Product_Category_1', 'Product_Category_2', 'Product_Category_3',
#-----------------------------charcater------------
#'Product_ID','Gender', 'Age','City_Category', 'Stay_In_Current_City_Years', 
       
#------------------------missing values_ train------------------------------------------
def missing(x):
    return sum(x.isnull())
friday_train.apply(missing,axis=0)

def counts(x):
    return x.value_counts()
counts(friday_train['Product_Category_2'])
counts(friday_train['Product_Category_3'])


friday_train['Product_Category_2']=friday_train['Product_Category_2'].fillna(value=8)
friday_train['Product_Category_3']=friday_train['Product_Category_3'].fillna(value=16)

#-------------------------------------------missing value-test--------------------------
friday_test.apply(missing,axis=0)

counts(friday_test['Product_Category_2'])
counts(friday_test['Product_Category_3'])

friday_test['Product_Category_2']=friday_test['Product_Category_2'].fillna(value=8)
friday_test['Product_Category_3']=friday_test['Product_Category_3'].fillna(value=16)


#-------------------------label encoding for object----------------
friday_train.dtypes
friday_test.dtypes

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
#--------------------------------train-----------------------------------
friday_train['Product_ID']= le.fit_transform(friday_train['Product_ID'])
counts(friday_train['Product_ID'])
friday_train['Gender']= le.fit_transform(friday_train['Gender'])
counts(friday_train['Gender'])
friday_train['Age']= le.fit_transform(friday_train['Age'])
counts(friday_train['Age'])
friday_train['City_Category']= le.fit_transform(friday_train['City_Category'])
counts(friday_train['City_Category'])
friday_train['Stay_In_Current_City_Years']= le.fit_transform(friday_train['Stay_In_Current_City_Years'])
counts(friday_train['Stay_In_Current_City_Years'])
#-----------------------------------test------------------------------------
friday_test['Product_ID']= le.fit_transform(friday_test['Product_ID'])
counts(friday_test['Product_ID'])
friday_test['Gender']= le.fit_transform(friday_test['Gender'])
counts(friday_test['Gender'])
friday_test['Age']= le.fit_transform(friday_test['Age'])
counts(friday_test['Age'])
friday_test['City_Category']= le.fit_transform(friday_test['City_Category'])
counts(friday_test['City_Category'])
friday_test['Stay_In_Current_City_Years']= le.fit_transform(friday_test['Stay_In_Current_City_Years'])
counts(friday_test['Stay_In_Current_City_Years'])

#-------------------------DV IDV-------------------------------------------
X= friday_train.drop('Purchase',axis=1)
Y= friday_train['Purchase']


#------------splitting train into x_train y_train x_test y_test----------------------

from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train= train_test_split(X,Y,test_size= 0.2)

#------------------------model------------------------------------------------
import statsmodels.api as sm
model_linear= sm.OLS(x_train,y_train).fit()
model_linear.summary()

#-------------------------linear regression-----------------------------------
from sklearn import linear_model
lm= linear_model.LinearRegression()
model_lm= lm.fit(x_train,y_train)
predicted_lm= model_lm.predict(x_train)
rmse_lm= np.sqrt(mean_squared_error(y_train,predicted_lm))
print(rmse_lm)
#---------------------------mean sqaure error--------------
from sklearn.metrics import mean_squared_error

#----------------------------------random forest----------------

from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=1000)
model_rf= rf.fit(x_train,y_train)
predicted_rf= model_rf.predict(x_train)
rmse_rf= np.sqrt(mean_squared_error(y_train,predicted_rf))
print(rmse_rf)


predicted_rf_x_test= model_rf.predict(x_test)
rmse_rf_x_test= np.sqrt(mean_squared_error(y_test,predicted_rf_x_test))
print(rmse_rf_x_test)

#----------------------------------support vector-------------------------------

from sklearn import svm
sv= svm.SVR(kernel='rbf')
model_sv= sv.fit(x_train,y_train)
predicted_sv= model_sv.predict(x_train)
rmse_sv= np.sqrt(mean_squared_error(y_train,predicted_sv))
print(rmse_sv)


#----------------------------MLP REGRESSOR----------------------------------------
from sklearn.neural_network import MLPRegressor
MLP= MLPRegressor(activation='relu',solver= 'adam',
                  max_iter=200, 
                  hidden_layer_sizes=(100,3))
MLP.fit(x_train,y_train)
predicted_mlp= MLP.predict(x_train)
rmse_MLP= np.sqrt(mean_squared_error(y_train,predicted_mlp))
print(rmse_MLP)

#--------------------------------test---------------------------------------------
predicted_test= model_rf.predict(friday_test)
output= pd.DataFrame(predicted_test)
output.to_csv('output.csv')

#--------------saving session----------------------------------------
# to load lod_session(finename)
import dill
filename = 'friday.pkl'
dill.dump_session(filename)
