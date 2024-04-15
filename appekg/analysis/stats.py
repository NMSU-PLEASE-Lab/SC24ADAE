#
# stats.py
#
# This script prints out statistics per tid/timemsec and pid/tid in a csv format. These statistics consist of min, max, std, mean, median, range, Q1, Q2, Q3, skew, kurtosis, count.
# To run the script, please use python3 and run 'python3 stats.py --help' for more instruction on how to run it.
# 
import math
import pandas as pd
import sys
from os import listdir
import numpy as np
import argparse 
import os
import json
from pathlib import Path
import scipy.stats as statss
from scipy.stats import shapiro
from scipy.stats import kstest
from statsmodels.stats.weightstats import ztest as ztest
import re
from collections import defaultdict

# Find files by extension
def list_files(dir_path, ext):
    files = listdir(dir_path)
    return [csv for csv in files if csv.endswith(ext)]

# Reference: https://medium.com/@pritul.dave/everything-about-moments-skewness-and-kurtosis-using-python-numpy-df305a193e46
def fun_skewness(arr):
    mean_ = np.mean(arr)
    median_ = np.median(arr)
    std_ = np.std(arr)
    if (std_ == 0):
        return np.nan
    skewness = 3*(mean_ - median_) / std_
    return skewness

# Reference: https://medium.com/@pritul.dave/everything-about-moments-skewness-and-kurtosis-using-python-numpy-df305a193e46
def fun_kurtosis(arr):
    mean_ = np.mean(arr)
    median_ = np.median(arr)
    mu4 = np.mean((arr - mean_)**4)
    mu2 = np.mean((arr-mean_)**2)
    if (mu2 == 0):
        return np.nan
    beta2 = mu4 / (mu2**2)
    gamma2 = beta2 - 3
    return gamma2

# extracte the unique thread IDs from the dataframe
def getThreadsID(df):
    threadsIDs = df.threadID.unique()
    threadsIDs.sort()
    return threadsIDs

# extracte the unique thread IDs from the dataframe
def getTimesecs(df):
    timesecs = df.timemsec.unique()
    timesecs.sort()
    return timesecs

# Print statistics function
def print_stats_per_thread(df):
    timesec = getTimesecs(df)
    threadsIDs = getThreadsID(df)
    print ("{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:}".format('timemsec','threadID','field','min','max','std','mean','median','range','Q1','Q2','Q3','skew','kurtosis','count'))
    for time in timesec:
        for thread in threadsIDs:
            _df = df[(df["threadID"] == thread) & (df["timemsec"] == time)].copy()
            _df.drop("timemsec", axis=1, inplace=True)
            _df.drop("threadID", axis=1, inplace=True)
            fields = _df.columns
            for f in fields:
                _count = _df.shape[0]
                _max = _df[f].max()
                _min = _df[f].min()
                _std = _df[f].std()
                _mean = _df[f].mean()
                _median = _df[f].median()
                _range = _max - _min
                _quantile1 = _df[f].quantile(q=0.25)
                _quantile2 = _df[f].quantile(q=0.5)
                _quantile3 = _df[f].quantile(q=0.75)
                _skew = fun_skewness(_df[f].tolist())
                _kurtosis = fun_kurtosis(_df[f].tolist())
                print ("{:},{:},{:},{:},{:3f},{:3f},{:3f},{:3f},{:3f},{:3f},{:3f},{:3f},{:3f},{:},{:}".format(time, thread, f, _min,_max,_std,_mean,_median,_range,_quantile1,_quantile2,_quantile3,_skew,_kurtosis,_count))


# Print statistics for each thread
def print_stats_per_process(df):
        print ("{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:},{:}".format('threadID','field','min','max','std','mean','median','range','Q1','Q2','Q3','skew','kurtosis','count'))
        threadsIDs = getThreadsID(df)
        for t  in threadsIDs:
            _df = df[df["threadID"] == t].copy()
            _df.drop("timemsec", axis=1, inplace=True)
            _df.drop("threadID", axis=1, inplace=True)
            fields = _df.columns
            for f in fields:
                _count = _df.shape[0]
                _max = _df[f].max()
                _min = _df[f].min()
                _std = _df[f].std()
                _mean = _df[f].mean()
                _median = _df[f].median()
                _range = _max - _min
                _quantile1 = _df[f].quantile(q=0.25)
                _quantile2 = _df[f].quantile(q=0.5)
                _quantile3 = _df[f].quantile(q=0.75)
                _skew = fun_skewness(_df[f].tolist())
                _kurtosis = fun_kurtosis(_df[f].tolist())
                print ("{:},{:},{:},{:},{:3f},{:3f},{:3f},{:3f},{:3f},{:3f},{:3f},{:3f},{:3f},{:}".format(t, f, _min,_max,_std,_mean,_median,_range,_quantile1,_quantile2,_quantile3,_skew,_kurtosis,_count))

def listSubDirectories(dir_path):
    dirPaths = []
    paths = {}
    for root, dirs, files in os.walk(dir_path, topdown=False):
        if not dirs:
            dirPaths.append(root)
            # print( os.path.basename(root))
            baseDir = Path(root).parts[-1]
            configDir = Path(root).parts[-2]
            if configDir not in paths.keys():
                paths[configDir] = {}
                paths[configDir]['jobs'] = []
                paths[configDir]['paths'] = []
            paths[configDir]['jobs'].append(baseDir)
            paths[configDir]['paths'].append(root)
    return(paths)

# process all thread files from all runs in a set of runs and generate an aggregate dataframe
def generateAggregateData(dir_path):
    df =pd.DataFrame()
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            if name.endswith(".csv"):
                _df = pd.read_csv(root + '/' + name)
                df = pd.concat([df, _df])
    if 'timemsec' in df.columns:
        df.drop("timemsec", axis=1, inplace=True)
    if 'threadID' in df.columns:
        df.drop("threadID", axis=1, inplace=True)
    # df = df.replace(0, np.NaN)
    return (df)

def generateCombinedMean(means):
    sum = sum(means)
    avg = sum/len(means)
    return avg

def generateCombinedStd(stds):
    var  = [i ** 2 for i in stds]
    print(var)
    varSum = sum(var)
    print(varSum)
    std = math.sqrt(varSum)
    return std

# generate statistics for a SET of runs
def generateAggregateStats(dfAll):
    if 'timemsec' in dfAll.columns:
        dfAll.drop("timemsec", axis=1, inplace=True)
    if 'threadID' in dfAll.columns:
        dfAll.drop("threadID", axis=1, inplace=True)
 
    fields = dfAll.columns
    # Conveer zero values to NaN, to be skipped when calculationg stats
    dfAll = dfAll.replace(0, np.NaN)
    stats = {}
    stats2 = []

    for field in fields:
        hbStats = {}
        hbStats2 = []
        
        mean = dfAll[field].mean()
        hbStats['mean'] = mean
        hbStats2.append(mean)
        max = dfAll[field].max()
        hbStats['max'] = max
        hbStats2.append(max)
        min = dfAll[field].min()
        hbStats['min'] = min
        hbStats2.append(min)
        std = dfAll[field].std()
        hbStats['std'] = std
        hbStats2.append(std)
        median = dfAll[field].median()
        hbStats['median'] = median
        hbStats2.append(median)
        range = max - min
        hbStats['range'] = range
        hbStats2.append(range)
        quantile1 = dfAll[field].quantile(0.25)
        hbStats['quantile1'] = quantile1
        hbStats2.append(quantile1)
        quantile2 = dfAll[field].quantile(q=0.5)
        hbStats['quantile2'] = quantile2
        hbStats2.append(quantile2)
        quantile3 = dfAll[field].quantile(q=0.75)
        hbStats['quantile3'] = quantile3
        hbStats2.append(quantile3)
        kurtosis = dfAll[field].kurtosis()
        hbStats['kurtosis'] = kurtosis
        hbStats2.append(kurtosis)
        skew = dfAll[field].skew()
        hbStats['skew'] = skew
        hbStats2.append(skew)
        skew = dfAll[field].count()
        hbStats['count'] = skew
        hbStats2.append(skew)
        stats[field] = hbStats
        stats2.append(hbStats2)
    
    # write stats in a json file
    # with open('result.json', 'w') as fp:
    #     json.dump(stats, fp, indent=4)
    return (stats,stats2)

def listSubDirectories(dir_path):
    rPaths = []
    for root, dirs, files in os.walk(dir_path, topdown=False):
        if not dirs:
            rPaths.append(root)
    return(rPaths)

def reduceData(inputPaths):
    means = []
    for p in inputPaths:
        df = pd.DataFrame()
        files = listdir(p)
        for file in files:
            if file.endswith('csv'):
                file = p + '/' + file
                df = pd.read_csv(file)
                df.drop("timemsec", axis=1, inplace=True)
                df.drop("threadID", axis=1, inplace=True)
            means.append(generateAggregateStats(df))
        if len(df.columns) != 0:
            fields = df.columns
    return(means,fields)

def reduceData2(inputPaths):
    means = []
    for p in inputPaths:
        df = pd.DataFrame()
        files = listdir(p)
        for file in files:
            avg = []
            if file.endswith('csv'):
                file = p + '/' + file
                df = pd.read_csv(file)
                df.drop("timemsec", axis=1, inplace=True)
                df.drop("threadID", axis=1, inplace=True)
                if len(df.columns) != 0:
                    fields = df.columns
                   
                for i in range(1,(int(len(fields)/2)+1)):
                    countList = df['hbcount' + str(i)].values.tolist()
                    countList = list(filter(lambda num: num != 0, countList))
                    countSum = np.sum(countList)
                    durationList = df['hbduration' + str(i)].values.tolist()
                    durationList = list(filter(lambda num: num != 0, durationList))
                    durationSum = np.sum(durationList)
                    avgDuration = durationSum/countSum
                    avg.append(avgDuration)
            means.append(avg)

            
            # means.append(generateAggregateStats(df))
        
    return(means)

def generateAggregateStats2(dfAll):
    # dfAll.drop("timemsec", axis=1, inplace=True)
    # dfAll.drop("threadID", axis=1, inplace=True)  
    fields = dfAll.columns
    # Conveer zero values to NaN, to be skipped when calculationg stats
    # dfAll = dfAll.replace(0, np.NaN)
    stats = []
    for field in fields:
        fieldList = dfAll[field].values.tolist()
        fieldList = list(filter(lambda num: num != 0.0, fieldList))
        print(field)
        print(np.sum(fieldList))
        mean = np.mean(fieldList)
        stats.append(mean)
    return(stats)


def isMeanSignificantlyDiff(o,m2,hb):
    if(m2 > (o[0] + 2*o[1]) or (m2 < o[0] - 2*o[1])):
        # print("{} is different".format(hb))
        return "different"
    else:
        # print("{} is not different".format(hb))
        return "not different"
# Parse arguments
parser = argparse.ArgumentParser(prog="stats.py", description='This script prints out statistics per tid/timemsec and per pid/tid in csv format. These statistics consist of min, max, std, mean, median, range, Q1, Q2, Q3, skew, kurtosis, count.')

# Define how a single command-line argument should be parsed.
parser.add_argument('--input', '-i', type=str, required=True, nargs='+', help="Input directory paths. This directory should contain only data of a single run.")
parser.add_argument('--type', '-t', type=str, required=False, choices=['per-tid-timemsec','per-pid-tid', 'all', 'per-config', 'per-pid'], help="Select what statistics to generate. Per threadID/timemsec or per processID/threadID.")
parser.add_argument('--stats', '-s', type=str, required=True, choices=['model','descriptive','t-test', 'compare'], help="Select what statistics to generate. descriptive or t-test")

# Create a new ArgumentParser object.
args = parser.parse_args()
dir_path = args.input 
type = args.type
stats = args.stats
# run = args.run

if stats == 't-test':
    # two inputs to compare the means of two populations
    if len(dir_path) != 2:
        parser.error("T-Test should be with two paths")
    df1 = generateAggregateData(dir_path[0])
    # reduce data of each pid to means, then compine data of all runs
    # runs1 = listSubDirectories(dir_path[0])
    # means,fields = reduceData(runs1)
    # df1 = pd.DataFrame(means, columns = fields)
    # df1 = df1.dropna()
    # print(df1)

    # compare data of the entire run (all pids) at once
    # combine data from all pids and then do t-test to comapre it with data of all good runs
    if type == None or type == 'all':
        # reduce data of each pid to means, then compine data of all runs
        # runs2 = listSubDirectories(dir_path[1])
        # means2,fields = reduceData(runs2)
        # df2 = pd.DataFrame(means2, columns = fields)
        # df2 = df2.dropna()
        # print(df2)
        df2 = generateAggregateData(dir_path[1])
        fields = df1.columns
        for field in fields:
            col_list = df2[field].values.tolist()
            # ignore 0 values
            col_list = list(filter(lambda num: num != 0, col_list))
            # print(np.sum(col_list))
            # print(len(col_list))
            col_list1 = df1[field].values.tolist()
            # ignore 0 values
            col_list1 = list(filter(lambda num: num != 0, col_list1))
            # print(np.sum(col_list1))
            # print(len(col_list1))
        
            # print(field)
            # print(kstest(df1[field], 'norm'))
            # print(kstest(df2[field], 'norm'))
            # print(shapiro(df1[field]))
            # print(shapiro(df2[field]))
            # result = ztest(df1[field], df2[field], value = 0)
            result = statss.ttest_ind(a=col_list, b=col_list1, equal_var=False, nan_policy='omit' )
            # print(statss.kruskal(df1[field],df2[field]))
            if result[1] < 0.05:
                print("{} is significantly different with p-value {}".format(field,result[1]))
            else:
                print("{} is not significantly different with p-value {}".format(field,result[1]))
    elif type == 'per-pid':
        csv_files = list_files(dir_path[1], ".csv")
        for csv in csv_files:
            csv_path = dir_path[1] + '/' + csv
            df2 = pd.read_csv(csv_path)
            fields = df1.columns
            print(csv)
            for field in fields:
                result = statss.ttest_ind(a=df1[field], b=df2[field], equal_var=False, nan_policy='omit')
                if result[1] < 0.05:
                    print("{} is significantly different with p-value {}".format(field,result[1]))
                else:
                    print("{} is not significantly different with p-value {}".format(field,result[1]))
                
# create stats model
if stats == 'model':
    # one input paths to the runs is required
    if len(dir_path) != 1:
        parser.error("one input path is required to create a model")
    dirs = listSubDirectories(dir_path[0])
    r = {}
    i = 0
    dics = []
    for dir in dirs:
        csv_files = list_files(dir, ".csv")
        dfn = pd.DataFrame()
        for csv in csv_files:
            csv_path = dir + '/' + csv
            _dfn = pd.read_csv(csv_path)
            dfn = pd.concat([dfn, _dfn])
        
        rrstats, runStat2 = generateAggregateStats(dfn)
        dics.append(runStat2)
    
    fields = dfn.columns
    mm = []
    for f in fields:
        mm.append([])
    for m in mm:
        for j in range(0,12):
            m.append([])
    
   # convert the list of stats per run to a list of stats metrics(means,max,std,..)
    for dic in dics:
        for i,  hb in enumerate(dic):
            for j,h in enumerate(hb):
                mm[i][j].append(h)

    rr = []
    # compute the comulative stats for each matric
    # mean of the means of the runs
    # mean of the std-dev of the runs (std-dev)
    for f, m in enumerate(mm):
        rr.append([])
        rr[f].append(np.mean(m[0]))
        rr[f].append(np.mean(m[3]))
    jsonData = {}
    for h,r in enumerate(rr):
        if (h+1)%2 != 0:
            # r.append("hbcount" + str((h+1)/2))
            hb = 'hbcount{}'.format(str(math.ceil((h+1)/2)))
            jsonData[hb] = r
        else:
        #    r.append("hbd" + str(math.floor((h+1)/2)))
            hb = 'hbduration{}'.format(str(int((h+1)/2)))
            jsonData[hb] =  r
    # save the model as a json file
    with open(os.getcwd() + '/' + 'model.json', 'w') as fp:
        json.dump(jsonData, fp, indent=4,default=int)
    
        

# comapre runs with the model
if stats == 'compare':
    # two input paths, the first one is to the 'model.json' file, and the second to the test runs
    if len(dir_path) != 2:
        parser.error("Descriptivr stats should be with two paths")
    # read the model
    df = pd.read_json(dir_path[0])
    dirs = listSubDirectories(dir_path[1])
    trueLabels = []
    predLabels = []
    allMetrics = []
    for dir in dirs:
        metrics = []
        if 'goodRuns' in dir:
            trueLabels.append('good')
        elif 'anomalousRuns' in dir:
            trueLabels.append("bad")

        csv_files = list_files(dir, ".csv")
        dfn = pd.DataFrame()
        for csv in csv_files:
            csv_path = dir + '/' + csv
            _dfn = pd.read_csv(csv_path)
            dfn = pd.concat([dfn, _dfn])
        # generate descriptive stats from a run
        rrstats3, runStat3 = generateAggregateStats(dfn)
        hbeats = dfn.columns
        hbmetric = []
        for t, hb in enumerate(hbeats):
            # check if mean of a hb is 2std greater or less than the mean of the model
            m = isMeanSignificantlyDiff(df[hb],runStat3[t][0],hb)
            hbmetric.append(m)
        allMetrics.append(hbmetric)
        if 'different' in hbmetric:
            predLabels.append('bad')
        else:
            predLabels.append('good')
        # print(runStat3)
   
    # confussion matrix
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    for i, label in enumerate(trueLabels):
        if label == 'bad' and predLabels[i] == 'bad':
            truePositive += 1
        elif label == 'bad' and predLabels[i] == 'good':
            falseNegative += 1
        elif label == 'good' and predLabels[i] == 'good':
            trueNegative += 1
        elif label == 'good' and predLabels[i] == 'bad':
            falsePositive += 1
    print("True Positive: {}".format(truePositive))
    print("False Positive: {}".format(falsePositive))
    print("True Negative: {}".format(trueNegative))
    print("False Negative: {}".format(falseNegative))
    
    allRunsResults = []
    for _ in dirs:
        allRunsResults.append([])
    for j, dir in enumerate(dirs):
        allRunsResults[j].append(dir)
        allRunsResults[j].append(trueLabels[j])
        allRunsResults[j].append(predLabels[j])
        for m in allMetrics[j]:
            allRunsResults[j].append(m)

    headers = ['run', 'trueLabel', 'predLabel']
    # extract hb to be used in the dataframe headers
    for h, _ in enumerate(allMetrics[0]):
        if (h+1)%2 != 0:
            hb = 'hbcount{}'.format(str(math.ceil((h+1)/2)))
            headers.append(hb)
        else:
            hb = 'hbduration{}'.format(str(int((h+1)/2)))
            headers.append(hb)

    # create a dataframe contains run path, true label, bad labels, and hb metrics
    df = pd.DataFrame(allRunsResults, columns = headers)
    # convert the dataframe to a csv file
    df.to_csv('my_csv.csv', index=False, header=True)
# descriptive stats of a run
if stats == 'descriptive':
    # only one input path to the run
    if len(dir_path) != 1:
        parser.error("Descriptivr stats should be with one path")
    df = generateAggregateData(dir_path[0])
    # get descriptive stats for all each pid in the input path (path of a single run)
    if type == None or type == "per-pid":
        csv_files = list_files(dir_path[0], ".csv")
        # Empty DataFrame
        df = pd.DataFrame()
        for csv in csv_files:
            csv_path = dir_path[0] + csv
            df = pd.read_csv(csv_path)
            runStat, rrstats = generateAggregateStats(df)
            # split file name to get the appekg-pid part to use in the result file name for each pid
            x = csv.split('.')
            # create a json file (containg the descriptive stats) in the same input path 
            with open(dir_path[0] + '/' + x[0] + '-result.json', 'w') as fp:
                json.dump(runStat, fp, indent=2, default=int)
    # get descriptive stats for all runs in the input path
    # it combines the data of all runs in a single df, the computes the stats form the df
    elif  type == 'all':
        statsPerRun = {}
        runStat, rrstatsd = generateAggregateStats(df)
            
        with open(os.getcwd() + '/' + '/result.json', 'w') as fp:
            json.dump(runStat, fp, indent=2, default=int)
    elif type == 'per-config':
        dirs = listSubDirectories(dir_path[0])
        r = {}
        for p in dirs:
            l = {}
            df = pd.DataFrame()
            for i, k in enumerate(dirs[p]['jobs']):
                _df = generateAggregateData(dirs[p]['paths'][i]) 
                df = pd.concat([df, _df])
                runStat = generateAggregateStats(_df)
                with open(dirs[p]['paths'][i] + '/result.json', 'w') as fp:
                    json.dump(runStat, fp, indent=4)
                l[dirs[p]['jobs'][i]] = runStat
            r[p] = l
            runStat = generateAggregateStats(df)
            with open(os.getcwd() + '/' + p + '/result.json', 'w') as fp:
                    json.dump(runStat, fp, indent=4)
    elif type == 'per-pid-tid':
        # Find CSV files
        csv_files = list_files(dir_path[0], ".csv")
        # Empty DataFrame
        df = pd.DataFrame()
        # # Merge CSV files and
        for csv in csv_files:
            csv_path = dir_path[0] + csv
            _df = pd.read_csv(csv_path)
            df = pd.concat([df, _df])
            # Print per processID/threadID 
            print(csv)
            print_stats_per_process(df)
    
    # Print per threadID/timemsec
    elif type == 'per-tid-timemsec':
        csv_files = list_files(dir_path[0], ".csv")
        # Empty DataFrame
        df = pd.DataFrame()
        for csv in csv_files:
            csv_path = dir_path[0] + csv
            _df = pd.read_csv(csv_path)
            df = pd.concat([df, _df])
        print_stats_per_thread(df)
    

# use t-test
if stats == 't-test':
    # two inputs to compare the means of two populations
    if len(dir_path) != 2:
        parser.error("T-Test should be with two paths")
    # df1 = generateAggregateData(dir_path[0])
    # reduce data of each pid to means, then compine data of all runs
    runs1 = listSubDirectories(dir_path[0])
    means = reduceData2(runs1)
    df1 = pd.DataFrame(means)
    df1 = df1.dropna()
    print(df1)

    runs2 = listSubDirectories(dir_path[1])
    means2 = reduceData2(runs2)
    df2 = pd.DataFrame(means2)
    df2 = df2.dropna()
    print(df2)

    # compare data of the entire run (all pids) at once
    # combine data from all pids and then do t-test to comapre it with data of all good runs
    if type == None or type == 'all':
        # reduce data of each pid to means, then compine data of all runs
        # runs2 = listSubDirectories(dir_path[1])
        # means2,fields = reduceData(runs2)
        # df2 = pd.DataFrame(means2, columns = fields)
        # df2 = df2.dropna()
        # print(df2)
    #     df2 = generateAggregateData(dir_path[1])
        fields = df1.columns
        for field in fields:
    #         col_list = df2[field].values.tolist()
    #         # ignore 0 values
    #         col_list = list(filter(lambda num: num != 0, col_list))
    #         # print(np.sum(col_list))
    #         # print(len(col_list))
    #         col_list1 = df1[field].values.tolist()
    #         # ignore 0 values
    #         col_list1 = list(filter(lambda num: num != 0, col_list1))
    #         # print(np.sum(col_list1))
    #         # print(len(col_list1))
        
    #         # print(field)
    #         # print(kstest(df1[field], 'norm'))
    #         # print(kstest(df2[field], 'norm'))
    #         # print(shapiro(df1[field]))
    #         # print(shapiro(df2[field]))
    #         # result = ztest(df1[field], df2[field], value = 0)
            result = statss.ttest_ind(a=df1[field], b=df2[field], equal_var=True, nan_policy='omit' )
    #         # print(statss.kruskal(df1[field],df2[field]))
            if result[1] < 0.05:
                print("{} is significantly different with p-value {}".format(field,result[1]))
            else:
                print("{} is not significantly different with p-value {}".format(field,result[1]))
    # elif type == 'per-pid':
    #     csv_files = list_files(dir_path[1], ".csv")
    #     for csv in csv_files:
    #         csv_path = dir_path[1] + '/' + csv
    #         df2 = pd.read_csv(csv_path)
    #         fields = df1.columns
    #         print(csv)
    #         for field in fields:
    #             result = statss.ttest_ind(a=df1[field], b=df2[field], equal_var=False, nan_policy='omit')
    #             if result[1] < 0.05:
    #                 print("{} is significantly different with p-value {}".format(field,result[1]))
    #             else:
    #                 print("{} is not significantly different with p-value {}".format(field,result[1]))

        
