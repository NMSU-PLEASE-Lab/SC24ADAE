#
# plot_per_interval.py 
#
# This script plots out stattistics(min, max, mean) per threadID and timesec, as well as per field and timesec. 
# To run the script, please use python3 and run 'python3 plot_per_interval.py --help' for more instruction on how to run it.
# 

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse 

def getThreadsID(df):
    threadsIDs = df.threadID.unique()
    threadsIDs.sort()
    return threadsIDs

def getTimesecs(df):
    timesecs = df.timemsec.unique()
    timesecs.sort()
    return timesecs

def getFields(df):
    fields = df.field.unique()
    fields.sort()
    return fields

# Plot by timesec and threadID function
def plot_timesec_threadID(threadsIDs, timesec, fields, outputPath, show):
    for f in fields:
        plt.figure()
        ax_= plt.gca()
        __df = df[df["field"] == f].copy()
        color_index = 0
        for t in threadsIDs:
            _df = __df[__df["threadID"] == t].copy()
            _df.plot(ax=ax_, x="timemsec", y="mean",  xlabel="Time [ms]", ylabel=f, label= "Thread " + str(t) , figsize=fig_size, color=line_colors[color_index % nColors])
            ax_.fill_between(x=_df["timemsec"], y1=_df["min"] , y2=_df["max"], alpha=.25, linewidth=0, color=shade_colors[color_index % nColors])
            color_index = color_index + 1
        ax_.legend(bbox_to_anchor=(1.0, 1.0))
        plt.savefig(str(f) + "_timesec_threadID.png", bbox_inches="tight", dpi=300)
        if show == "true" or show == "True": 
            plt.show()

# Plot by timesec and field function
def plot_timesec_filed(timesec, fields, outputPath, show):
    for f in fields:
        plt.figure()
        ax_= plt.gca()
        _df = df[df["field"] == f].copy()
        color_index = 0
        _df = _df.groupby(["timemsec"], as_index=False).mean(numeric_only=True)
        _df.plot(ax=ax_, x="timemsec", y="mean",  xlabel="Time [ms]", ylabel=f , figsize=fig_size, color=line_colors[color_index % nColors])
        ax_.fill_between(x=_df["timemsec"], y1=_df["min"] , y2=_df["max"], alpha=.25, linewidth=0, color=shade_colors[color_index % nColors])
        color_index = color_index + 1
        ax_.legend(bbox_to_anchor=(1.0, 1.0))
        plt.savefig(str(f) + "_timesec_field.png", bbox_inches="tight", dpi=300)
        if show == "true" or show == "True": 
            plt.show()

# Set colors
line_colors =  ["#fc0f03", "#104E8B", "#FFD700", "#595959", "#FF69B4", "#FF8000", "#66CDAA", "#FFE4C4", "#98F5FF","	#68228B"]
shade_colors = ["#fc2403", "#1E90FF", "#CDAD00", "#7F7F7F", "#CD6090", "#CD8500", "#7FFFD4", "#CDB79E", "#7AC5CD", "#B23AEE"]
nColors = len(line_colors)

# Figure settings
fig_size = (8,4.5)
x_label = "Time (sec)"


# Parse arguments
parser = argparse.ArgumentParser(prog="plot_per_interval", description='Test passing arguments')

# Define how a single command-line argument should be parsed.
parser.add_argument('--input', '-i', type=str, required=True, help="Input file path. Full path to the per threadID/timemsec output csv file.")
parser.add_argument('--output', '-o', type=str, required=False, help="Output directory path. If not defined, it will be saved in a location where scripts run.")
parser.add_argument('--show', '-s', type=str, required=False, help="Preview plots. Default is set to false.", default=False)
parser.add_argument('--plot', '-p', type=str, required=False, choices=['per-tid-timemsec','per-field-timemsec'], help="Select what plots to generate. Per threadID/timemsec or per field/timemsec. Default is both.", default=True)

# Create a new ArgumentParser object.
args = parser.parse_args()
inputPath = args.input
outputPath = args.output + '/'
show = args.show
plot = args.plot

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Read CSV
df = pd.read_csv(inputPath)
timesec = getTimesecs(df)
threadsIDs = getThreadsID(df)
fields = getFields(df)

# Plot data
if plot == 'per-tid-timemsec':
    plot_timesec_threadID(threadsIDs, timesec, fields, outputPath, show)
elif plot == 'per-field-timemsec':
    plot_timesec_filed(timesec, fields, outputPath, show)
else:
    plot_timesec_threadID(threadsIDs, timesec, fields, outputPath, show)
    plot_timesec_filed(timesec, fields, outputPath, show)