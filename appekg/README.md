# AppEKG

AppEKG is a heartbeat instrumentation library and analysis framework,
enabling the tracking and analysis of application performance at the
application phase level with a low-overhead, production-usable 
method.

The defined macros are the preferred way of creating AppEKG
instrumentation in an application, but the underlying functions
are also available in the API; the functions for begin/end
heartbeats do not inherently support a rate factor, however. 

The macro interface is:  
- EKG_BEGIN_HEARTBEAT(id, rateFactor) 
- EKG_END_HEARTBEAT(id) 
- EKG_PULSE_HEARTBEAT(id, rateFactor) 
- EKG_INITIALIZE(numHeartbeats, samplingInterval, appid, jobid, rank, silent) 
- EKG_FINALIZE() 
- EKG_DISABLE() 
- EKG_ENABLE() 
- EKG_NAME_HEARTBEAT(id, name) 
- EKG_IDOF_HEARTBEAT(name) 
- EKG_NAMEOF_HEARTBEAT(id) 

Heartbeat IDs are small integers starting at 1, and must be sequential.
A unique heartbeat ID is meant to represent a particular phase or kernel 
of the application, and generally each instrumentation site has a unique
heartbeat ID.

_rateFactor_ controls the speed of heartbeat production, if an instrumentation
site is invoked too frequently. A rateFactor of 100, for example, would 
produce a heartbeat once every 100 executions of the instrumentation site.

AppEKG initialization accepts as parameters the number of unique heartbeats
(maximum heartbeat ID), the number of seconds between data samples, a
unique application ID, job ID, MPI rank, and a silent flag for turning
off (unlikely but possible) stderr messages. If the job ID is left 0, PBS
and Slurm environment variables are checked and used to set it. An 
application ID is useful when creating a historical database of multiple
applications' data.

Heartbeats can be given a name using the API; names should generally
refer to the conceptual meaning of the application phase or kernel the
heartbeat is capturing.

Environment Variables:  
- APPEKG_SAMPLING_INTERVAL : integer, number of seconds between samples; 
                             will override AppEKG initialization parameter
- APPEKG_OUTPUT_PATH : string to prepend to output file names; '/' is added 
                       at end and does not need to be included. If string
                       contains one %d, the application ID is inserted at
                       that spot; if it contains two %d's, the job ID is
                       inserted for the second one. Other % options will 
                       cause the string to be used as is, without 
                       substitutions.
- PBS_JOBID : if found, used for the 'jobid' data field, if param jobid=0
- SLURM_JOB_ID : if found, used for the 'jobid' data field, if param jobid=0

## Building

_make_ should work in the main directory; _make doc_ will create doxygen
documentation. If you need to select optional output modes such as LDMS
streams, edit the Makefile to set it up properly.

In the _test_ directory, _make_ will build a variety of tests. The test 
_evensum_ is the most complete, using OpenMP threads to generate heartbeats
per thread. Run as 'OMP_NUM_THREADS=2 ./evensum' to ,e.g., set the number 
of threads to 2.

## Running an application

In the _test_ directory, running the _evensum_ executable will produce two
output files one named 'appekg-###.json' and one named 'appekg-###.csv'. 
The number in the filename is the PID of the process (on a cluster, each
process will produce its own heartbeat data files).

The JSON file is the metadata for the heartbeat data, and includes a variety
of data fields; most are self-explanatory. The field _component_ is a number
that is extracted from the host name, if the name has a number in it (most HPC
clusters set up their computational nodes with names with ID numbers in them).

The CSV files is the heartbeat data, in column format. The first two columns
are a timestamp (end of sampling interval, milliseconds since the beginning 
of the execution) and a thread ID (a unique, small but non-consecutive integer
value). The rest of the columns is heartbeat data, two columns per heartbeat;
the first is the number of heartbeats that occurred in this sampling interval,
the second is the average duration of these heartbeats (microseconds).

AppEKG does keep heartbeat data per thread, using hashing to quickly locate
the thread data region; hash collisions can cause threads to be ignored, and
the hash table is static in size and will not grow. The default size is 57 but
can be changed by editing _appekg.h_ (around line 90). 

## Running analyses

There are different scripts in the _analyses_ directory that you can run to 
conduct different analyses over the collected heartbeat data. 

### Requirements

The analyses scripts depend on the following modules:
- pandas>=1.5.0
- numpy>=1.23.3
- matplotlib>=3.6.0
- argparse>=1.4.0

__Note__: These are not necessarily hard requirements, other new but less recent 
versions will probably work; we have not done extensive version testing.
 
To install those requirements in your environment using _pip_, run:
```shell
pip3 install -r requirements.txt
```

### Running statistics on heartbeat data

The _stats.py_ prints out statistics per threadID per timemsec and processID per threadID in a CSV format. 
These statistics consist of min, max, std, mean, median, range, Q1, Q2, Q3, skew, 
kurtosis, count. It reads the CSV files that were produced by the AppEKG tool. 

**How to run the script?**
```shell
python3 stats.py [options]
```
 **Options:**
- --input or -i: required to specify heartbeat data path. Make sure the 
heartbeat data in the the directory is from a single run.
- --type or -t: required to select what to data statistics to output. 
    - per-tid-timemsec - statistics calculated over all ranks per threadID per timemsec.
    - per-pid-tid - statistics calculated per rank per threadID.


Examples:
1. To print statistics based on heartbeat data located in "/path/toMyInput" per rank per threadID
   in "/path/toMyInput", run:
```shell
python3 stats.py --input "/path/toMyInput" --type per-pid-tid
```
2. To save statistics based on heartbeat data located in "/path/toHBData/" over all ranks per threadID per timemsec and
   save the data in "/path/where/toSaveStats/stats.csv"
```shell
python stats.py -i /path/toHBData/ -p per-tid-timemsec > /path/where/toSaveStats/stats.csv
```

### Plotting the counts and durations of all heartbeats

The _plotting.py_ script plots heartbeat counts and durations. It reads the 
CSV and JSON files that were produced by the AppEKG tool and then plots the 
heartbeat data. It plots heartbeat data per rank and per threadID per rank. 
Each created plot contains a plot of each heartbeat. the AppEKG tool produces a CSV file from each rank, where each file consists the heartbeat data of the related rank. If an application is multithreaded, different threads in each CSV will present, where each thread produces its own heartbeat data.
1. Per rank plotting: this plots the heartbeat data of the CSV files (ranks) regardless of the threadIDs. It skips the threadIDs and consider the heartbeat data of a single rank for plotting.
2. Per threadID per rank plotting: this plots the heartbeat data of each threadID. When the application is multithreaded, each CSV file contains the heartbeat data per threadID. If there are 4 threads, there will be 2 plots per threadID (one for heartbeat count and one for duration). This means threre will be 8 plots per CSV file (rank).  

**How to run the script?**
```shell
python3 plotting.py [options]
```
 **Options:**
- --input or -i: required to specify heartbeat data path. Make sure the 
heartbeat data in the the directory is from a single run.
- --output or -o: optional to specify where to save the plots. If not defined, 
the input path is used.
- --plot or -p: optional to select what to plot. If not defined all plot types 
are produced (per rank and per threadID per processID).
    - per-rank: plot heartbeat data per-processID only. If the app were run on 4 processes, 8 plots are created. (4 heartbeat counts and 4 heartbeat durations).
    - per-tid-rank: plot heartbeat data per-threadID per-processID. If the app were run on 4 threads and 2 processes, 16 plots are created (8 heartbeat counts and 8 heartbeat durations), where each plot represents the data per-threadID per-processID.
- --tid or -t: optinal to plot data of specific threadIDs. If not defined, heartbeat data of all threadIDs will plot
- --rank or -r: optional to plot data of specific ranks. If not defined, heartbeat data of all ranks will plot 
- --getranktid or -g: optional to print thread IDs and ranks. Default is set to false.
- --show: optional to preview plots. Default is set to false.

Examples:
1. To plot heartbeat data located in "/path/toMyInput" per rank and save plots
   in "/path/toMyInput", run:
```shell
python3 plotting.py --input "/path/toMyInput" --plot per-rank
```
2. To plot heartbeat data located in "/path/toHBData/" per threadID per rank and
   save the plots in "/path/where/toSavePlots"
```shell
python plotting.py -i /path/toHBData/ -o /path/where/toSavePlots -p per-tid-rank
```
3. To plot heartbeat data located in "/path/toMyInput" of rank 15 only per rank
   and save plots in the same path of input, run:
```shell
python3 plotting.py --input "/path/toMyInput" --plot per-rank --rank 15
```

### Plotting the counts and durations of all heartbeats per threadID/timemsec and field/timemsec

The _plot_per_interval.py_ script plots heartbeat counts and durations. It reads the CSV file
file that was produced by the _stats.py_ script when chosing an "per-tid-timemsec" option and 
then plots the average heartbeat data over all processes. It plots heartbeat data per threadID 
and timemsec, and per field and timemsec.

**How to run the script?**
```shell
python3 plot_per_interval.py [options]
```
 **Options:**
- --input or -i: required to specify data path for output of the _stats.py_ script when chosing an "per-tid-timemsec" option.
- --output or -o: optional to specify where to save the plots. If not defined, 
the input path is used.
- --plot or -p: optional to select what to plot. If not defined all plot types 
are produced (per threadID per timemsec and per field per timemsec).
    - per-tid-timemsec: plot heartbeat data per-threadID per-timemsec only. If the app has 3 heartbeats, 6 plots are created. (3 heartbeat counts and 3 heartbeat durations).
    - per-field-timemsec: plot average heartbeat data for all processes per-field per-timemsec. If the app has 3 heartbeats, 6 plots are created. (3 heartbeat counts and 3 heartbeat durations).
- --show: optional to preview plots. Default is set to false.

Examples:
1. To plot heartbeat data located in "/path/toMyInput" per threadID per timemsec and save plots
   in "/path/toMyInput", run:
```shell
python3 plot_per_interval.py --input "/path/toMyInput" --plot per-tid-timemsec
```
2. To plot heartbeat data located in "/path/toHBData/" per field per timemsec and
   save the plots in "/path/where/toSavePlots"
```shell
python plot_per_interval.py  -i /path/toHBData/ -o /path/where/toSavePlots -p per-field-timemsec
```
## Acknowledgments

This work was supported in part by Sandia National Laboratories.
No endorsement is implied.

