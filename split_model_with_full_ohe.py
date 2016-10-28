# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:46:20 2016

@author: bingl

Idea of the split model: since there are so many categorical features, and the 
data set is so large(~180000 data points). A straightforward linear regression 
model would be inefficient to represent all the details. Since we have a large 
data set, we could do way more than this naive regression method. First, we
should split the data, or say cluster the points into different groups according
to their categorical features, and then fit each group to an independent small
model. This method needs a reasonable split, say after splitting, each small
model should get enough data, like order of ~1000. Thus a rough estimate would
be 180000/1000 ~ 180 ~ 2 ^ (7~8). for example, we could do a hierarchical binary
split with about 7 layers. The key point is that after clustering, each independent
model must be more interpretable. 

Second consideration would be about the use of categorical features, i.e. whether
take it as a categorical (split into different models), or take it as a numerical
feature by converting it into numerical by something like "labelencoder" as
shown as below. So the difficulty here would be how can we recognize which type 
each feature belongs to and Can we run a learning algorithm to find the optimal 
solution.

Can we try Greedy algorithm: in the categorical space, split the space into 2
step by step, and in each split we try to maximize the linearization character
of the 2 clusters, i.e. better fit by linear/or other regression models, and at 
the same time, use the clustering of correlation matrix as a reference

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn import svm

data = pd.read_csv("C:\\python_project\\Kaggle_allstate\\train.csv")
print data.head(5)
plt.close()

#encode categorical features
contFts = []
allFts = []
for colName, x in data.drop(["id","loss"], axis=1).iloc[1,:].iteritems():
    allFts.append(colName)
    if(not str(x).isalpha()):
        contFts.append(colName)
catFts = list(set(allFts)-set(contFts))

for f in catFts:    
    le = LabelEncoder()
    le.fit(data[f].unique())
    data[f] = le.transform(data[f])
       
sample_split = []

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        
                        sample = data[(data['cat1']==i) & (data['cat1']==i) &
                                      (data['cat2']==j) & (data['cat2']==j) &
                                      (data['cat23']==k) & (data['cat23']==k) &
                                      (data['cat4']==l) & (data['cat4']==l) &
                                      (data['cat5']==m) & (data['cat5']==m) &
                                      (data['cat6']==n) & (data['cat6']==n) ]
                        
                        sample_split.append(sample)
                        
n_split = len(sample_split)

size_split = []
splits_over_thousand = [i for i in range(n_split) 
                      if ((len(sample_split[i])>1000) 
                      and (len(sample_split[i])<10000))]
                      
print "number of splits that have more than 1000 and less than 10000 samples: \n", len(splits_over_thousand)
print splits_over_thousand

num = sum([len(sample_split[i]) for i in splits_over_thousand])
print "total samples covered by these chosen splits: \n", num

otherFts = list(set(allFts)-set(['cat1', 'cat2', 'cat4', 'cat5', 'cat6', 'cat23']))
otherCatFts = list(set(catFts)-set(['cat1', 'cat2', 'cat4', 'cat5', 'cat6', 'cat23']))
features = data[otherFts]

count = 0
sample_sizes = []
sample_squared_error = []

for sample in sample_split:
    
    print "************************************\n"
    print "group # : ", count
    count += 1
    size_sample = sample.shape[0]
    print size_sample
    if (size_sample > 10000 or size_sample == 0):
        continue
    
    sample_cont = sample[contFts]
    sample_y = np.log(sample["loss"])
    sample_cat_transformed = []
    
    for i, col in enumerate(otherCatFts):
        n_cats = max(sample[col])+1
        onehot_encoder = OneHotEncoder(sparse=False,n_values=n_cats)
        feature = onehot_encoder.fit_transform(sample[col].values.reshape(size_sample, 1))
        sample_cat_transformed.append(feature)

    encoded_cats = np.column_stack(sample_cat_transformed)
    sample_encoded = np.concatenate((encoded_cats,sample_cont.values),axis=1)
    if (size_sample > 1000 ):        
        regr = linear_model.LinearRegression()
    else:
        regr = linear_model.Ridge(alpha = 10.0)
    
    plt.figure(count)
    regr.fit(sample_encoded, sample_y)
    pred_y = regr.predict(sample_encoded)
    plt.scatter(pred_y, sample_y)
    print("Mean squared error: %.8f" % (sum((pred_y - sample_y) ** 2)/size_sample))
    print("mean_absolute_error %s" % mean_absolute_error(pred_y, sample_y))
    
    sample_sizes.append(size_sample)
    sample_squared_error.append(sum((pred_y - sample_y) ** 2)/size_sample)
    
    plt.xlabel("predictions")
    plt.ylabel("real logloss")
    x = np.linspace(5.,11., 100)
    y = x
    plt.xlim([5., 11.5])
    plt.ylim([4., 11.5])
    plt.plot(x, y, c= 'red', linewidth = 2.0)
    plt.show()

print sum(np.array(sample_squared_error)*np.array(sample_sizes))/sum(np.array(sample_sizes))
