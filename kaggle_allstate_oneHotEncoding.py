# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:29:45 2016

"""

import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import xgboost as xgb
#from sklearn import svm

train = pandas.read_csv("C:\\python_project\\Kaggle_allstate\\train.csv", dtype={'id': np.int32})
test = pandas.read_csv("C:\\python_project\\Kaggle_allstate\\test.csv", dtype={'id': np.int32})

datasize = 10000
print train.head(3)
print train.describe()

# 116 columns of categorical features, plus 14 continuous features plus log loss
# First try with partial data to speed up the learning process
train = train.drop(['id'], axis=1).iloc[:datasize,:]
train["loss"] = numpy.log1p(train["loss"])
test = test.drop(['id'], axis=1).iloc[:datasize,:]

split = 116
train_cont = train.iloc[:,split:]
train_cat = train.iloc[:, :split]

test_cont = test.iloc[:,split:]
test_cat = test.iloc[:, :split]
cats = train_cat.columns

# now labels contain all possible categories for each feature from 1 to 116
labels = {}
for i, col in enumerate(cats):
    label_in_train = train[col].unique()
    label_in_test = test[col].unique()
    labels[col] = list(set(label_in_train) | set(label_in_test))

#One hot encode all categorical attributes
train_cat_transformed = []
test_cat_transformed = []

for i, col in enumerate(cats):
    print i, col, len(labels[col])
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[col])
    
    feature = label_encoder.transform(train_cat.iloc[:,i]).reshape(train_cat.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[col]))
    feature = onehot_encoder.fit_transform(feature)
    train_cat_transformed.append(feature)

    feature = label_encoder.transform(test_cat.iloc[:,i]).reshape(test_cat.shape[0], 1)
    feature = onehot_encoder.fit_transform(feature)
    test_cat_transformed.append(feature)
    
encoded_cats = numpy.column_stack(train_cat_transformed)
train_encoded = numpy.concatenate((encoded_cats,train_cont.values),axis=1)
encoded_cats = numpy.column_stack(test_cat_transformed)
test_encoded = numpy.concatenate((encoded_cats, test_cont.values),axis=1)

#scale the X feature for both training set and testing set
X_train = train_encoded[:, :-1]
Y_train = train_encoded[:, -1]
X_test = test_encoded

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print "after the one-hot-encoding, the number of all features is: \n"
print encoded_cats.shape[1]
del train
del train_cont
del encoded_cats
del test
del test_cont

#get the number of rows and columns
r, c = train_encoded.shape
model = LinearRegression(n_jobs=-1)
model.fit(X_train, Y_train)

Y_test_pred = model.predict(X_test)
Y_train_pred = model.predict(X_train)
result = np.mean((Y_train - Y_train_pred)**2)
print("average square error (training set) %s" % result)
print("mean_absolute_error %s" % mean_absolute_error(Y_train, Y_train_pred))

plt.figure(1)
plt.scatter(Y_train_pred, Y_train, c = 'b')
plt.xlabel("predictions")
plt.ylabel("real logloss")
plt.xlim([5., 11.5])
plt.ylim([4., 11.5])
x = numpy.linspace(5.,11., 100)
y = x
plt.plot(x, y, c= 'red', linewidth = 2.5)
#plt.savefig("C:\\python_project\\Kaggle_allstate\\final\\LR_OHE_with_Scaling.png")
plt.show()


'''
X = X_train[:8000, :]
Y = Y_train[:8000]
X_val = X_train[8000:, :]
Y_val = Y_train[8000:]

early_stopping = 25
params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = "rmse"
params['eta'] = 0.1
params['gamma'] = 0.5290
params['min_child_weight'] = 4.2922
params['colsample_bytree'] = 0.3085
params['subsample'] = 0.9930
params['max_depth'] = 7
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 1001

d_train = xgb.DMatrix(X, label = Y)
d_valid = xgb.DMatrix(X_val, label = Y_val)
watchlist = [(d_train, 'train'), (d_valid, 'eval')]

clf = xgb.train(params,
                d_train,
                100000,
                watchlist,
                early_stopping_rounds=early_stopping)
                
plt.figure(2)
y_pred = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
plt.scatter(y_pred, Y_val, c = 'b')
plt.xlabel("predictions")
plt.ylabel("real logloss")
plt.xlim([5., 11.5])
plt.ylim([4., 11.5])
x = numpy.linspace(5.,11., 100)
y = x
plt.plot(x, y, c= 'red', linewidth = 2.5)
plt.savefig("C:\\python_project\\Kaggle_allstate\\final\\LR_OHE_plus_BOOSTTREE_val.png")
result = np.mean((y_pred - Y_val)**2)
print("average square error (training set) %s" % result)
print("mean_absolute_error %s" % mean_absolute_error(y_pred, Y_val))

plt.show()

plt.figure(3)
y_pred = clf.predict(d_train, ntree_limit=clf.best_ntree_limit)
plt.scatter(y_pred, Y, c = 'b')
plt.xlabel("predictions")
plt.ylabel("real logloss")
plt.xlim([5., 11.5])
plt.ylim([4., 11.5])
x = numpy.linspace(5.,11., 100)
y = x
plt.plot(x, y, c= 'red', linewidth = 2.5)
plt.savefig("C:\\python_project\\Kaggle_allstate\\final\\LR_OHE_plus_BOOSTTREE_train.png")
result = np.mean((y_pred - Y)**2)
print("average square error (training set) %s" % result)
print("mean_absolute_error %s" % mean_absolute_error(y_pred, Y))

plt.show()
'''