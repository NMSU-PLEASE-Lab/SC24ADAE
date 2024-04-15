
#
# This script plots heartbeat counts and durations. It reads the CSV and JSON files that were
# produced by the AppEKG tool and then plots the heartbeat data. It plots heartbeat data per 
# rank and per threadID per rank. Each created plot contains a plot of each heartbeat. The script
# reads heartbeat data from the CSV files and metadata from the JSON files. Each input directory should 
# hold the data of a single run.
# Requirements:
#   pandas>=1.5.0
#   numpy>=1.23.3
#   matplotlib>=3.6.0
#   argparse>=1.4.0
# 
# How to install requirements?
#   pip3 install -r requirements.txt
# 
# How to run the script?
#   python3 plotting.py [options]
#  
# Options:
#   --input or -i: required to specify heartbeat data path. Make sure the heartbeat data in the
#     the directory is from a single run.
#   --output or -o: optional to specify where to save the plots. If not defined, the input path
#     is used.
#   --plot or -p: optional to select what to plot. If not defined all plot types is produced 
#     (per processID, per threadID per rank).
#     per-rank: plot heartbeat data per rank only. If the app were run on 4 processes, 8 plots
#     are created. (4 heartbeat counts and 4 heartbeat durations).
#     per-tid-rank: plot heartbeat data per-threadID per-rank. If the app were run on 4 threads 
#     and 2 processes, 16 plots are created (8 heartbeat counts and 8 heartbeat durations), where each
#     plot represents the data per-threadID per-processID.
#   --tid or -t: optinal to plot data of specific threadIDs. If not defined, heartbeat data of all 
#     threadIDs will plot
#   --rank or -r: optional to plot data of specific ranks. If not defined, heartbeat data of all 
#     ranks will plot 
#   --getranktid or -g: optinal to print thread IDs and ranks. Default is set to false.
#   --show: optional to preview plots. Default is set to false.
# Examples:
#   1. python3 plotting.py --input "/path/toMyInput" --type per-rank
#     To plot heartbeat data located in "/path/toMyInput" per processID and save plots in 
#     "/path/toMyInput".
#   2. python -i /path/toHBData/ -o /path/where/toSavePlots -t per-tid-rank
#     To plot heartbeat data located in /path/toHBData/ per threadID per rank and save the plots in
#     "/path/where/toSavePlots".
# 
# 
#

import pandas as pd
import sys
from os import listdir
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import json
import argparse 

# globals
showPlot = ""
colors = ['r', 'g', 'b', 'y', 'k','c','m']

#---------------------------------------------------------------------
# find files by extension
#---------------------------------------------------------------------
def list_files(dir_path, fileExt):
    if not(os.path.exists(dir_path)):
        print("Input dir {} not exist".format(dir_path))
        exit(1)
    files = listdir(dir_path)
    return [csv for csv in files if csv.endswith(fileExt)]

#---------------------------------------------------------------------
# convert csv to a dataframe
#---------------------------------------------------------------------
def extractDataFromCSV(dataFile):    
    file2 = open(dataFile,"r")
    df = pd.read_csv(file2)
    return df

#---------------------------------------------------------------------
# extracte the unique thread IDs from a dictionary
#---------------------------------------------------------------------
def extractTHreadIDS(ranks):
    threadIDs = []
    for pid in ranks.keys():
        for tid in ranks[pid]["tid"]:
            threadIDs.append(tid)
    return threadIDs

#---------------------------------------------------------------------
# convert a json file to a dictionary
#---------------------------------------------------------------------
def readJsonFile(jsonFile):
    data = json.load(jsonfile)
    return data

#---------------------------------------------------------------------
# get number of HBs and HB names form a dataframe
#---------------------------------------------------------------------
def getNoHBs(df):
    noOfHBs = 0
    hbNames = {}
    for col in df.columns:
        if "hbcount" in col:
            noOfHBs += 1
            hbNames[noOfHBs] = "hbeat" + str(noOfHBs)
    return noOfHBs, hbNames

#---------------------------------------------------------------------
# get number heartbeats and heartbeat names
#---------------------------------------------------------------------
def createHBNames(path, files, df):
    hbNames = {}
    noOfHBs = 0
    processIDs = []
    ranks = {}
    # get number of heartbeats and heartbeat names from json file if not empty,
    # and from csv file otherwise
    data = json.load(open(path + files[0], 'r'))
    if data["hbnames"]:
        for hbId in data["hbnames"].keys():
            noOfHBs += 1
            hbNames[hbId] = data["hbnames"][hbId]
    else:
        noOfHBs, hbNames = getNoHBs(df)
    return(noOfHBs, hbNames)

# def getRanks(ranksArgs, path, files):
#     ranks = {}
#     for file in files:
#         data = json.load(open(path + file, 'r'))
#         ranks[data["pid"]] = data["rank"]
#         # processIDs.append(data["pid"])
#         # ranks.append(data["rank"])
#     if ranksArgs != None:
#         key_list = list(ranks.keys())
#         val_list = list(ranks.values())
#         rr = {}
#         newRanks = ranksArgs.split(',')
#         for i in newRanks:
#             i = int(i)
#             if i in list(ranks.values()):
#                 position = val_list.index(i)
#                 key = key_list[position]
#                 rr[key] = i
#             else:
#                 print("Rank {} is not valid, please select valid ranks".format(i))
#                 exit(1)
#         ranks = rr
#     return(ranks)

#---------------------------------------------------------------------
# get the unique thread IDs
#---------------------------------------------------------------------
def getThreadIDs(tidArgs):
    tids = tidArgs.split(',')
    tids = [int(i) for i in tids]
    return(tids)

#---------------------------------------------------------------------
# plot heartbeat duration 
#---------------------------------------------------------------------
def plotHBDuration(df, numofHBs, hbNames, rank, threadID, outPath, plotType):
    i = 1
    fig=plt.figure(figsize=(7.2,4.45))
    for i in range(1,numofHBs+1):
        plt.plot(df["timemsec"], df["hbduration" + str(i)],colors[i-1],label = hbNames[str(i)])
        # for i,func in enumerate(hbNames):
        #     plt.plot(df[headers2[i]],colors[i],label = func)
        # Set the y-axis scale log base 10
        #plt.ylim(bottom=0.1) # not sure if we need this on time plots
    plt.yscale("log",base = 10)
    plt.xlabel("Time (sec)",fontsize=12)
    plt.ylabel("Average Time (milisec)",fontsize=12)
    # position the legend
    plt.legend(loc='best',prop={'size':12})
    # x and y ticks font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.legend(bbox_to_anchor=(0.9, 0.69))
    # Save the figure as png
    if plotType == "per-rank":
        plt.title(label="Heartbeat Durations Rank ({})".format(rank))
        plt.savefig(outPath + 'HBDuration-Rank{}.png'.format(rank), bbox_inches="tight", dpi=100)
    elif plotType == "per-tid-rank":
        plt.title(label="Heartbeat Durations Rank ({}), ThreadID ({})".format(rank, threadID))
        plt.savefig(outPath + 'HBDuration-Rank{}-ThreadID{}.png'.format(rank, threadID), bbox_inches="tight", dpi=100)
    elif plotType == "per-tid":
        plt.title(label="Heartbeat Durations ThreadID ({})".format(threadID))
        plt.savefig(outPath + 'HBDuration-ThreadID{}.png'.format(threadID), bbox_inches="tight", dpi=100)
    # preview if specified in the arguments --show
    if showPlot:
        plt.show()

#---------------------------------------------------------------------
# get the unique ranks
#---------------------------------------------------------------------
def getRanks(path, files):
    ranks = {}
    for file in files:
        r = {}
        jdata = json.load(open(path + file, 'r'))
        csvdata = extractDataFromCSV("appekg-" + str(jdata["pid"]) + ".csv")
        r["pid"] = jdata["pid"]
        r["tid"] = list(csvdata.threadID.unique())
        ranks[jdata["rank"]] = r
    return ranks

#---------------------------------------------------------------------
# plot heartbeat count 
#---------------------------------------------------------------------
def plotHBCount(df, numofHBs, hbNames, rank, threadID, outPath, plotType):
    fig=plt.figure(figsize=(7.2, 4.45))
    for i in range(1,numofHBs + 1):
        plt.plot(df["timemsec"], df["hbcount" + str(i)],colors[i-1], label = hbNames[str(i)])
        # for i,func in enumerate(hbNames):
        #     plt.plot(df[headers2[i]],colors[i],label = func)
        # Set the y-axis scale log base 10
    plt.ylim(bottom=0.1)
    plt.yscale("log", base = 10)
    plt.xlabel("Time (msec)", fontsize=12)
    plt.ylabel("Interval Heartbeat Count", fontsize=12)
    # position the legend
    # plt.legend(bbox_to_anchor=(0.55, 0.7))
    # position the legend
    plt.legend(loc='best',prop={'size':12})
    # x and y ticks font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Save the figure as png
    if plotType == "per-rank":
        plt.title(label="Heartbeat Count Rank ({})".format(rank))
        plt.savefig(outPath + 'HBCount-Rank{}.png'.format(rank), bbox_inches="tight", dpi=100)
    elif plotType == "per-tid-rank":
        plt.title(label="Heartbeat Count Rank ({}), ThreadID ({})".format(rank, threadID))
        plt.savefig(outPath + 'HBCount-Rank{}-threadID{}.png'.format(rank, threadID),bbox_inches="tight",dpi=100)
    elif plotType == "per-tid":
        plt.title(label="Heartbeat Count ThreadID ({})".format(threadID))
        plt.savefig(outPath + 'HBCount-threadID{}.png'.format(threadID),bbox_inches="tight",dpi=100)
    # preview if specified in the arguments --show
    if showPlot:
        plt.show()

#---------------------------------------------------------------------
# plot hbcount and hbtime per threadID (each threadID has 2 figures, 
# one hbcount and one for hbtime)
#---------------------------------------------------------------------
def plotPerTID(path, ranks, numofHBs, hbNames, outPath):
    for pid in ranks.keys():
        df =  extractDataFromCSV(path + "appekg-" + str(pid) + ".csv")
        df = pd.concat([df, df])
        rank = pid
        threadIDs = extractTHreadIDS(df)
    for j in threadIDs:
        plotHBCount(df.loc[df['threadID'] == j], numofHBs, hbNames, rank, j, outPath, "per-tid")
        plotHBDuration(df.loc[df['threadID'] == j], numofHBs,hbNames, rank, j, outPath, "per-tid")

#---------------------------------------------------------------------
# plot hbcount and hbtime per threadID per rank
#---------------------------------------------------------------------
def plotPerTIDPerPID(path, ranks, numofHBs, hbNames, outPath, argsTid):
    threadIDs = extractTHreadIDS(ranks)
    for pid in ranks.keys():
        df =  extractDataFromCSV(path + "appekg-" + str(ranks[pid]["pid"]) + ".csv")
        rank = pid
        if argsTid != None:
            tids = argsTid.split(',')
            for i in tids:
                i = int(i)
                if i not in threadIDs:
                    print("ThreadID {} is not used, please select valid thread IDs!".format(i))
                    exit(1)
                else:
                    plotHBCount(df.loc[df['threadID'] == i], numofHBs, hbNames, rank, i, outPath, "per-tid-rank")
                    plotHBDuration(df.loc[df['threadID'] == i], numofHBs,hbNames, rank, i, outPath, "per-tid-rank")
        else:
            for j in ranks[pid]["tid"]:
                plotHBCount(df.loc[df['threadID'] == j], numofHBs, hbNames, rank, j, outPath, "per-tid-rank")
                plotHBDuration(df.loc[df['threadID'] == j], numofHBs,hbNames, rank, j, outPath, "per-tid-rank")

#---------------------------------------------------------------------
# extract the rank of each process ID from a dictionary
#---------------------------------------------------------------------
def extractRanks(ranks):
    r = []
    for pid in ranks.keys():
        r.append(pid)
    return r

#---------------------------------------------------------------------
# plot hbcount and hbtime for all ranks(each rank has 2 figures, 
# one hbcount and one for hbtime)
#---------------------------------------------------------------------
def plotPerPID(path, ranks, numofHBs, hbNames, outPath, argsRank):
    r = extractRanks(ranks)
    if argsRank != None:
        pids = argsRank.split(',')
        for i in pids:
            i = int(i)
            if i not in r:
                print("Rank {} is not used, please select valid rank!".format(i))
                exit(1)
            else:
                df =  extractDataFromCSV(path + "appekg-" + str(ranks[i]["pid"]) + ".csv")
                plotHBCount(df, numofHBs, hbNames, i, -1, outPath, "per-rank")
                plotHBDuration(df, numofHBs,hbNames, i, -1,outPath, "per-rank")
    else:
        for pid in ranks.keys():
            df =  extractDataFromCSV(path + "appekg-" + str(ranks[pid]["pid"]) + ".csv")
            rank = pid
            plotHBCount(df, numofHBs, hbNames, rank, -1, outPath, "per-rank")
            plotHBDuration(df, numofHBs,hbNames, rank, -1,outPath, "per-rank")

#---------------------------------------------------------------------
# print thread IDs and ranks
#---------------------------------------------------------------------
def printTIDandRanks(ranks):
    print("Rank \t ThreadID")
    for key in ranks.keys():
        print("{} \t".format(key), end = "")
        for value in ranks[key]["tid"]:
            if value == ranks[key]["tid"][-1]:
                print(value)
            else:
                print("{}, ".format(value), end = "")       
    
    
#---------------------------------------------------------------------
# main
#---------------------------------------------------------------------
parser = argparse.ArgumentParser(prog="plotting", description = 'Plotting Heartbeat Data')
# define how a single command-line argument should be parsed.
parser.add_argument('--input', '-i', type=str, required=True, help = "Heartbeat data path. Each path should have the heartbeat data of a single run.")
parser.add_argument('--plot', '-p', type=str, required=False, choices = ['per-rank', 'per-tid-rank'], help="Select what plots to generate. Per processID, Per ThreadID or per Per ThreadID Per ProcessID. Default is all.")
parser.add_argument('--output', '-o', type=str, required=False, help = "Output directory path. If not defined, plots will be saved in the input path.")
parser.add_argument('--getranktid', '-g', type=str, required=False, help = "Print thread IDs and ranks. Default is set to false.", default=False)
parser.add_argument('--tid', '-t', type=str, required=False, help = "Specify threadID to plot. If not defined, heartbeat data of all threadIDs will plot.")
parser.add_argument('--rank', '-r', type=str, required=False, help = "Specify rank to plot. If not defined, heartbeat data of all ranks will plot.")
parser.add_argument('--show', '-s', type=str, required=False, help="Preview plots. Default is set to false.", default=False)


# create a new ArgumentParser object
args = parser.parse_args()
showPlot = args.show
inputPath = args.input 
if not inputPath.endswith('/'):
    inputPath += '/'
if not os.path.exists(inputPath):
    print("No such directory, use a valid input path.")
if args.output == None:
    outputPath = inputPath
else:
    outputPath = args.output
    if not outputPath.endswith('/'):
        outputPath += '/'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
# find csv files 
csvFiles = list_files(inputPath, ".csv")
# find json files
jsonFiles = list_files(inputPath, ".json")
# exit if no heartbeat data
if not jsonFiles or not csvFiles:
    print("The input directory has no input files")
    exit(1)
# select one of the csv files to extract # of heartbeats and heartbeat names
ranks = getRanks(inputPath, jsonFiles)
df = extractDataFromCSV(inputPath + csvFiles[0])
numofHBs,hbNames = createHBNames(inputPath, jsonFiles, df)
pltType = args.plot
if args.getranktid in ["True", "true"]:
    printTIDandRanks(ranks)
# find threadIDs of the first rank
rank1 = next(iter(ranks))
threadIDs = list(ranks[rank1]["tid"])
# plot data (per-rank, per-tid or per-tid-rank)
if pltType == "per-rank":
    plotPerPID(inputPath, ranks, numofHBs, hbNames, outputPath, args.rank)
# elif pltType == "per-tid":
    # print("per-tid")
    # plotPerTID(inputPath, ranks, numofHBs, hbNames, outputPath)
elif pltType == "per-tid-rank":
    # don't plot per-tid per-rank if the application is not multithreaded,
    # we can only plot per-rank
    if len(threadIDs) > 1:
        plotPerTIDPerPID(inputPath, ranks, numofHBs, hbNames, outputPath, args.tid)
    else:
        print("The application is not multithreaded, you can only plot per rank!")
else:
    if len(threadIDs) > 1:
        plotPerPID(inputPath, ranks, numofHBs, hbNames, outputPath,args.rank)
        plotPerTIDPerPID(inputPath, ranks, numofHBs, hbNames, outputPath, args.tid)
    else:
        print("The application is not multithreaded, you can only plot per rank!")
        plotPerPID(inputPath, ranks, numofHBs, hbNames, outputPath, args.rank)