
#
#  The main goal of this script is to build a classification model for APPEKG data (time series data) by a well-known time series classification technique (shapelets). 
# It first extracts the data from the APPEKG runs.
# Then it pads the time series with lengths < longest one with zeros to make all-time series the same length.
# The script reduces time series dimensionality using Piecewise Aggregate Approximation. 
# Reference: https://tslearn.readthedocs.io/en/latest/gen_modules/piecewise/tslearn.piecewise.PiecewiseAggregateApproximation.html#tslearn.piecewise.PiecewiseAggregateApproximation
# Then, it applies LearningShapelets from tslearn module to find the most representative shapelets.
# Reference: https://tslearn.readthedocs.io/en/latest/gen_modules/shapelets/tslearn.shapelets.LearningShapelets.html#tslearn.shapelets.LearningShapelets
#  

from os import listdir
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import csv
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from pyts.approximation import PiecewiseAggregateApproximation
# from pyts.classification import LearningShapelets
from tslearn.shapelets import LearningShapelets
from tslearn.utils import to_time_series_dataset
from pyts.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

# It reads the output file of each run, to check if a run is good or anomalous.
# The output files contain a line specifying the label of the run (good run or anomalous run)
def getLabelFromOutput(run):

    # for pennant 
    if 'APPEKG' in run:
        label = 'good'
    elif 'anomalousRuns' in run:
        label = 'anomalous'
    files = os.listdir(run)
    for fileName in files:
        if fileName.endswith('out'):
            filePath = run + '/' + fileName
            f1 = open(filePath, "r")
            lines2 = f1.readlines()
            for line in lines2:
                if 'good run' in line:
                    label = 'good'
                elif 'anomalous run' in line:
                    label = 'anomalous'
            break
    return label

# find number of hbs. APPEKG prints out hbnames in a json file
def getNumberHBs(dir_path):
    found = False
    for run in dir_path:
        #root, _ = os.path.split(root)
        files = os.listdir(run)
        for fileName in files:
            if fileName.endswith(".json"):
                fname = run + '/' + fileName
                print(fname)
                with open(fname) as fp:
                    data = json.load(fp)
                    numberHB = len(data["hbnames"])
                    found = True
                break
        if found:
            break

    return numberHB



# process all thread files and generate an aggregate dataframe
# it finds values of hbcounts and durations and thier means of each csv file in a run
# it returns a list of lists for (hbcount_i, hbduration_i, hbcount_i means, hbduration_i mean), i is hb Id.
def generateAggregateData(dir_path):
    numHB = getNumberHBs(dir_path)
    hbCounts = []
    hbDurations = []
    hbCountMeans = []
    hbDurMeans = []
    for i in range(numHB):
        hbCounts.append([])
        hbDurations.append([])
        hbCountMeans.append([])
        hbDurMeans.append([])
    
    df =pd.DataFrame()
    labels = []
    for run in dir_path:
        #root, _ = os.path.split(root)
        files = os.listdir(run)
        label = getLabelFromOutput(run)
        for fileName in files:
            if fileName.endswith(".csv"):
                
                df =pd.DataFrame()
                fname = run + '/' + fileName
                df = pd.read_csv(fname)
                for d in df['threadID'].unique():
                    labels.append(label)

                for hb in range(numHB):
                    for d in df['threadID'].unique():
                        
                        x = df.loc[df['threadID'] == d, "hbcount" + str(hb+1)].tolist()
                        xnew = [k for k in x if k != 0.0]
                        xmean = sum(xnew)/len(xnew)
                        hbCountMeans[hb].append(xmean)
                        j = df.loc[df['threadID'] == d,"hbduration" + str(hb+1)].tolist()
                        jnew = [k for k in j if k != 0.0]
                        jmean = sum(jnew)/len(jnew)
                        hbDurMeans[hb].append(jmean)
                        hbCounts[hb].append(x)
                        hbDurations[hb].append(j)

    if 'timemsec' in df.columns:
        df.drop("timemsec", axis=1, inplace=True)
    # if 'threadID' in df.columns:
    #     df.drop("threadID", axis=1, inplace=True)
    # df = df.replace(0, np.NaN)
    return (hbCounts, hbDurations, labels, hbCountMeans, hbDurMeans)

# List good and anomalous APPEKG runs
# The data dir contains 2 sub dirs (APPEKG and anomalousRuns). It skips the clean runs (no APPEKG)
# It returns list of paths of runs
def listSubDirectories(dir_path):
    rPaths = []
    for root, dirs, files in os.walk(dir_path, topdown=False):
        if not dirs:
            # skip clean runs (no APPEKG)
            if "cleanRuns" in root:
                continue
            if "newAnalysis" in root:
                continue
            if "ICs" in root:
                continue
            # CoMD runs have a folder used for input
            if "galaxy" in root or "parameterfiles" in root or "ICs" in root or "CoMD" in root:
                root, _ = os.path.split(root)
            rPaths.append(root)
    return(rPaths)

# change data to fit shapelet learning algorithm (all time sieres will same length)
# it bads all time sieres < len of longest time series with 0s
def makeDataFitShapelet(hbData):
    hbData = to_time_series_dataset(hbData)
    for t in hbData:
        t[np.isnan(t)] = 0
    return hbData

# Split arrays or matrices into random train and test subsets.
def splitData(data, labels, testSize):
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size = testSize, random_state=42)
    return X_train, X_test, y_train, y_test

# reduce the time series dimensionality from n to m (n_segments) using 
# Piecewise Aggregate Approximation. first divide the original time-series 
# into M equally sized frames (segments) and then compute the mean values for each frame. 
# The sequence assembled from the mean values is the PAA approximation (i.e., transform) 
# of the original time-series.
def reduceTimeSieriesLength(XTrain, XTest, segments):
    paa = PiecewiseAggregateApproximation(window_size=segments)
    XTrain = paa.fit_transform(XTrain)
    XTest = paa.fit_transform(XTest)
    return XTrain, XTest



from sklearn.model_selection import cross_val_score

def find_optimal_shapelets(X, y, num_shapelets_range, shapelet_size_range, cv):
    """
    Finds the optimal number and size of shapelets using cross-validation.

    Parameters:
    - X: numpy array, shape (n_samples, n_timestamps, n_features)
        Input time series data.
    - y: numpy array, shape (n_samples,)
        Target labels.
    - num_shapelets_range: list or range
        Range of numbers of shapelets to try.
    - shapelet_size_range: list or range
        Range of shapelet sizes to try.
    - cv: int, default=5
        Number of folds for cross-validation.

    Returns:
    - optimal_num_shapelets: int
        Optimal number of shapelets.
    - optimal_shapelet_size: int
        Optimal shapelet size.
    """
    best_score = -np.inf
    optimal_num_shapelets = None
    optimal_shapelet_size = None
    allScores = []

    for num_shapelets in num_shapelets_range:
        for shapelet_size in shapelet_size_range:
            shapelet_clf = LearningShapelets(n_shapelets_per_size={shapelet_size: num_shapelets})
            scores = cross_val_score(shapelet_clf, X, y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            allScores.append(mean_score)

            if mean_score > best_score:
                best_score = mean_score
                optimal_num_shapelets = num_shapelets
                optimal_shapelet_size = shapelet_size


    return optimal_num_shapelets, optimal_shapelet_size, allScores 

# Example usage:
# optimal_num_shapelets, optimal_shapelet_size = find_optimal_shapelets(X_train, y_train, 
#                                                                        num_shapelets_range=range(1, 6), 
#                                                                        shapelet_size_range=range(3, 8), 
#                                                                        cv=5)
# print("Optimal number of shapelets:", optimal_num_shapelets)
# print("Optimal shapelet size:", optimal_shapelet_size)

# Compute number and length of shapelets for learning shapelets
# Reference: https://tslearn.readthedocs.io/en/latest/gen_modules/shapelets/tslearn.shapelets.grabocka_params_to_shapelet_size_dict.html
# It returns Dictionary giving, for each shapelet length, the number of such shapelets to be generated
def getNumberSizeShapelets(XTrain,YTrain):
    n_ts, ts_sz = XTrain.shape[:2]
    n_classes = len(set(YTrain))
    # Set the number of shapelets per size as done in the original paper
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                       ts_sz=ts_sz,
                                                       n_classes=n_classes,
                                                       l=0.2,
                                                       r=1)
    return shapelet_sizes


def learningShapelets(XTrain, YTrain, shapeletSizes, iter):
    t0 = time.time()
    shpClf = LearningShapelets(n_shapelets_per_size=0.1,random_state=42, tol=0.01)
    shpClf.fit(XTrain, YTrain)
    t1 = time.time()
    totalTime = t1 - t0
    return shpClf, totalTime

def computeTimeSeriesMeans(data):
    means = []
    for d in data:
        d = d[d!=0]
        mean = np.mean(d)
        means.append(mean)
    return means

def findTimeseriesShapeletDis(nshapelets, XTrain, distances, df):
    dis = []
    for l in range(nshapelets):
        dis.append([])
    for k, _ in enumerate(XTrain):
        for j in range(nshapelets):
            dis[j].append(distances[k][j])
    for h, diss in enumerate(dis):
        df['disShapelet'+ str(h+1)] = diss
    

def plotShapeletTimeSeriesDis(df, nshapeltes, hbname):
    labels = df["labels"]
    y = df[hbname + 'Mean']
    for i in range(nshapeltes):
        x = df['disShapelet' + str(i+1)]
        xGood = []
        yGood = []
        xBad = []
        yBad = []
        for b, l in enumerate(labels):
            if l == 'good':
                xGood.append(x[b])
                yGood.append(y[b])
            else:
                xBad.append(x[b])
                yBad.append(y[b])
    
        plt.scatter(xGood,yGood, color = 'g', label = 'GoodRun')
        plt.scatter(xBad,yBad, color='r', label = 'BadRun')
        plt.xlabel('d(x,s' + str(i+1) + ')')
        plt.ylabel(hbname)
        plt.legend()
        plt.title(hbname + 'Time Series Distance to Shapeplet' + str(i+1))
        plt.savefig(hbname + 'shapelet' + str(i+1) + '.png')
        plt.close()

def findMostInfluentialShapelets(shWeights):
    shId = 0
    max = 0
    for i, sh in enumerate(shWeights[0]):
        if(np.max(sh) > max):
            max = np.max(sh)
            shId = i+1
    return shId


# main

if len(sys.argv) < 3:
    print('error: missing the runs path\nPlease use: python3 {} </path/to/app/runs> <application name>'.format(sys.argv[0]))
    exit(0)

path =  sys.argv[1]  
appName = sys.argv[2]
dirPath = listSubDirectories(path)
hbcounts, hbdurations, labels, hbCountMeans, hbDurationMeans = generateAggregateData(dirPath)

# get number of hbeats
hbN = getNumberHBs(dirPath)

# find shapelets for each hb (count and duration)
for i in range(hbN):
    # hbcount shapelets
    hbCountDf =pd.DataFrame()
    hbDurDf =pd.DataFrame()
    hbCountData = hbcounts[i]
    hbCountData = makeDataFitShapelet(hbCountData)
    hbCountData = hbCountData.reshape(hbCountData.shape[0], hbCountData.shape[1])
    X_train1, X_test1, y_train1, y_test1 = splitData(hbCountData, labels, 0.7)
    X_train1, X_test1 = reduceTimeSieriesLength(X_train1, X_test1, 2)
    print(X_train1.shape)

    means = computeTimeSeriesMeans(X_train1)
    hbCountDf['hbCount' + str(i+1) + "Mean"] = means
    X_train1 = MinMaxScaler().fit_transform(X_train1)
    print(X_train1.shape)
    hbname = 'hbCount' + str(i+1)
    X_test1 = MinMaxScaler().fit_transform(X_test1)
    
    hbCountDf['labels'] = y_train1
    optimal_num_shapelets, optimal_shapelet_size, scores = find_optimal_shapelets(X_train1, y_train1, 
                                                                       num_shapelets_range=[1, 2, 3, 4], 
                                                                       shapelet_size_range=[50, 100, 150, 200], 
                                                                       cv=2)
    print(hbname)
    print("optimal_num_shapelets ", optimal_num_shapelets)
    print("optimal_shapelet_size ", optimal_shapelet_size)
    print("scores: ", scores)
    # shapeletSizes = getNumberSizeShapelets(X_train1, y_train1)
    # print(X_train1.shape)
    # hbCountShpClf, time1 = learningShapelets(X_train1, y_train1, shapeletSizes, 800)
    # # distances = hbCountShpClf.transform(X_train1)
    # # nshapelets = len(hbCountShpClf.shapelets_)
    # # ypred = hbCountShpClf.predict(X_train1)
    # # hbCountDf['predicted'] = ypred
    # # findTimeseriesShapeletDis(nshapelets, X_train1, distances, hbCountDf)
    # # hbname = 'hbCount' + str(i+1)
    # # plotShapeletTimeSeriesDis(hbCountDf, nshapelets, hbname)
    # # outPutFile = 'hbCount' + str(i+1) + 'shapelets.xlsx'
    # # hbCountDf.to_excel(outPutFile)
    # print(time1)
    # print("HBCount" + str(i+1) + ":",hbCountShpClf.score(X_test1, y_test1))
    # shapelet_weights = hbCountShpClf.get_weights("shapelets_0_0")
    # id = findMostInfluentialShapelets(shapelet_weights)
    # print(id)

    # hbduration shapelets
    hbname2 = 'hbDuration' + str(i+1)
    hbDurData = hbdurations[i]
    hbDurData = makeDataFitShapelet(hbDurData)
    hbDurData = hbDurData.reshape(hbDurData.shape[0], hbDurData.shape[1])
    X_train2, X_test2, y_train2, y_test2 = splitData(hbDurData, labels, 0.70)
    
    X_train2, X_test2 = reduceTimeSieriesLength(X_train2, X_test2, 2)
    
    means2 = computeTimeSeriesMeans(X_train2)
    X_train2 = MinMaxScaler().fit_transform(X_train2)
    X_test2 = MinMaxScaler().fit_transform(X_test2)
    hbDurDf['hbDuration' + str(i+1) + "Mean"] = means2
    hbDurDf['labels'] = y_train2
    optimal_num_shapelets, optimal_shapelet_size, scores = find_optimal_shapelets(X_train2, y_train2, 
                                                                       num_shapelets_range=[1, 2, 3], 
                                                                       shapelet_size_range=[30, 50, 100, 200], 
                                                                       cv=2)
    print(hbname2)
    print("optimal_num_shapelets ", optimal_num_shapelets)
    print("optimal_shapelet_size ", optimal_shapelet_size)
    print("scores: ", scores)

    # shapeletSizes2 = getNumberSizeShapelets(X_train2, y_train2)
    # hbDurShpClf, time2 = learningShapelets(X_train2, y_train2, shapeletSizes2, 800)
    # print(hbDurShpClf.coef_[0])
    # shapelet_size = hbDurShpClf.shapelets_[0]
    # print(hbDurShpClf.shapelets_[0])
    # distances2 = hbDurShpClf.transform(X_train2)
    # ypred2 = hbDurShpClf.predict(X_train2)
    # hbDurDf['predicted'] = ypred2
    # print(time2)
    # print("HBDuration" + str(i+1) + ":",hbDurShpClf.score(X_test2, y_test2))
    # nshapelets = len(hbDurShpClf.shapelets_)
    # findTimeseriesShapeletDis(nshapelets, X_train2, distances2, hbDurDf)
    # hbname2 = 'hbDuration' + str(i+1)
    # plotShapeletTimeSeriesDis(hbDurDf, nshapelets, hbname2)
    # outPutFile = appName + 'hbDuration' + str(i+1) + 'shapelets.xlsx'
    # hbDurDf.to_excel(outPutFile)
    


