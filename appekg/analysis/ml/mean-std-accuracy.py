#
# This script randomly chooses 1 good/bad run (the rest of the data is considered as the training dataset)
# and test accuracy for different classification algorithms.
# 
#
# To run this script:
# python mean-std-accuracy.py PATH_GOOD_RUNS PATH_BAD_RUNS NUM_OF_HEARTBEATS TYPE_OF_TEST_DATA
#
# ex: python mean-std-accuracy.py /PATH_TO_DATA/miniFE/good/ /PATH_TO_DATA/miniFE/bad/ 5 good
#
# Note:
# This script uses both mean and std values for calculation
# To test only using mean, comment out line 85 and 87
# To test only using std, comment out line 84 and 86
#

import pandas as pd
import numpy as np
import os
import sys
import random
from fnmatch import fnmatch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

def get_file_list(path, pattern):
    list_path = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                list_path.append(os.path.join(path, name))

    return list_path

def separate_test_batch(list_runs, pattern):
    path_test_run = os.path.dirname(random.choice(list_runs))
    list_runs = [i for i in list_runs if not i.startswith(path_test_run)]
    list_test_run = get_file_list(path_test_run, pattern)
    return list_runs, list_test_run

def get_mean_std_list(file_list, performance, num_heartbeats):

    column_names = list()
    column_names.append('threadID')

    for c in range(0, num_heartbeats):
        column_names.append('hbc' + str(c+1) + '-mean')
        column_names.append('hbc' + str(c+1) + '-std')
        column_names.append('hbd' + str(c+1) + '-mean')
        column_names.append('hbd' + str(c+1) + '-std')

    df_performance = pd.DataFrame(columns=column_names)
    
    for index, filename in enumerate(file_list):
        df = pd.read_csv(filename)
        # drop timemsec column
        df = df.drop(columns=['timemsec'])
        # drop all 0 rows
        df = df[df.loc[ : , df.columns!='threadID'].sum(axis=1) > 0]
        # calculate mean
        data_list = []
        data_list.append(str(df['threadID'].iloc[0]))

        for i in range(0, num_heartbeats):
            data_list.append(round(df["hbcount" + str(i+1)].mean(), 4))
            data_list.append(round(df["hbcount" + str(i+1)].std(), 4))
            data_list.append(round(df["hbduration" + str(i+1)].mean(), 6))
            data_list.append(round(df["hbduration" + str(i+1)].std(), 6))
        
        # write mean-std values in another dataframe
        df_performance.loc[len(df_performance.index)] = data_list

    # mark all rows as good or bad
    df_performance['performance'] = performance
    
    return df_performance
    
def all_classifiers_train_test_split(df, preprocessing, num_heartbeats):

    feature_columns = list()
    for c in range(0, num_heartbeats):
        feature_columns.append('hbc' + str(c+1) + '-mean')
        feature_columns.append('hbc' + str(c+1) + '-std')
        feature_columns.append('hbd' + str(c+1) + '-mean')
        feature_columns.append('hbd' + str(c+1) + '-std')

    X = df[feature_columns] # Features
    y = df.performance # Target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
    
    if preprocessing == 1:
        # Standarize
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
    elif preprocessing == 2:
        # Normalize
        X_train = Normalizer().transform(X_train)
        X_test = Normalizer().transform(X_test)
    
    models = {}

    models['Logistic Regression'] = LogisticRegression()
    models['Support Vector Machines'] = LinearSVC()
    models['Decision Trees'] = DecisionTreeClassifier()
    models['Random Forest'] = RandomForestClassifier()
    models['Naive Bayes'] = GaussianNB()
    models['K-Nearest Neighbor'] = KNeighborsClassifier()
    
    accuracy, precision, recall = {}, {}, {}
    
    for key in models.keys():
        models[key].fit(X_train, y_train)
        predictions = models[key].predict(X_test)
        # Calculate accuracy
        accuracy[key] = accuracy_score(predictions, y_test)
    
    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy'])
    df_model['Accuracy'] = accuracy.values()

    print(df_model)

def all_classifiers_real_test(df_train, df_test, preprocessing, num_heartbeats):
    feature_columns = list()
    for c in range(0, num_heartbeats):
        feature_columns.append('hbc' + str(c+1) + '-mean')
        feature_columns.append('hbc' + str(c+1) + '-std')
        feature_columns.append('hbd' + str(c+1) + '-mean')
        feature_columns.append('hbd' + str(c+1) + '-std')

    X_train = df_train[feature_columns] # Features
    y_train = df_train.performance # Target variable
    
    X_test = df_test[feature_columns] # Features
    y_test = df_test.performance # Target variable
    
    if preprocessing == 'standard-scaler':
        # Standarize
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
    elif preprocessing == 'normalizer':
        # Normalize
        X_train = Normalizer().transform(X_train)
        X_test = Normalizer().transform(X_test)
    
    models = {}

    models['Logistic Regression'] = LogisticRegression()
    models['Support Vector Machines'] = LinearSVC()
    models['Decision Trees'] = DecisionTreeClassifier()
    models['Random Forest'] = RandomForestClassifier()
    models['Naive Bayes'] = GaussianNB()
    models['K-Nearest Neighbor'] = KNeighborsClassifier()
    
    accuracy, precision, recall = {}, {}, {}
    
    for key in models.keys():
        models[key].fit(X_train, y_train)
        predictions = models[key].predict(X_test)
        # Calculate accuracy
        accuracy[key] = accuracy_score(predictions, y_test)
    
    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy'])
    df_model['Accuracy'] = accuracy.values()

    print(df_model)


inputPath = sys.argv[1]
outputPath = sys.argv[2]

#app_name = 'miniFE' # (not using) 
#path_good_run = '/Users/sudippodder/Documents/RA_Works/ml-scripts/dataset-hist/good'
#path_bad_run = '/Users/sudippodder/Documents/RA_Works/ml-scripts/dataset-hist/bad'
#num_heartbeats = 5 # number of HBs of the application
#test_type = good # good or bad
#preprocessing = 1 # 0 = no-preprocessing, 1 = standard-scaler, 2 = normalizer

path_good_run = sys.argv[1]
path_bad_run = sys.argv[2]
num_heartbeats = int(sys.argv[3]) # number of HBs of the application
test_type = sys.argv[4] # good or bad

pattern = "*.csv"

list_goodruns = get_file_list(path_good_run, pattern)
list_badruns = get_file_list(path_bad_run, pattern)

df_test = pd.DataFrame()
list_test_run = []

if test_type == 'good':
    # separate test batch from good runs
    list_goodruns, list_test_run = separate_test_batch(list_goodruns, pattern)
    performance = 1
    df_test = get_mean_std_list(list_test_run, performance, num_heartbeats)
else:
    # separate test batch from bad runs
    list_badruns, list_test_run = separate_test_batch(list_badruns, pattern)
    performance = 0
    df_test = get_mean_std_list(list_test_run, performance, num_heartbeats)

df_good = get_mean_std_list(list_goodruns, 1, num_heartbeats)
df_bad = get_mean_std_list(list_badruns, 0, num_heartbeats)
df_train = df_good.append(df_bad, ignore_index=True)

print()
print('Sample Size: ')
print('Train, Test = (' + str(df_train.shape[0]) + ', ' + str(df_test.shape[0]) + ')')

print()
print('.------------.-------------.')
print('****** Mean Std Data *******')
print('.------------.-------------.')

print()
print('----- No Preprocessing -----')
all_classifiers_real_test(df_train, df_test, 0, num_heartbeats)
print()
print('----- Standard Scaler -----')
all_classifiers_real_test(df_train, df_test, 1, num_heartbeats)
print()
print('----- Normalizer -----')
all_classifiers_real_test(df_train, df_test, 2, num_heartbeats)


