
# This Python script is designed to calculate quantitative quality values and gather
# descriptive statistics for various applications. It evaluates the quality value based 
# on the formula:
# quality = problemSize /(#threads × runTime)
# where the quality is adjusted for the number of threads used and the runtime for processing. 
# Each application's folder is expected to contain two subfolders: APPEKG, which holds data 
# from various configurations of APPEKG runs, and anomalousRuns, which stores data from runs 
# with anomalous configurations.
# The script generates two spreadsheets: one containing metadata, quality values, labels,
# and descriptive statistics of the runs; the other, a statistical model based on these stats. 
# To evaluate the quality matrix and descriptive statistics for a specific application, you should 
# specify the application name with the -a flag and the path to the application's folder. 
# For instance, to assess the quality values and descriptive statistics for all Lammps runs,
# you would execute:
# python3 /path/to/jobQuality.py -a lammps -p /path/to/appRuns
# appRuns directory contains the following subdirectories:
# - APPEKG
# - anomalousRuns
# - misConfig


import argparse 
import re
import pandas as pd
import os
import numpy as np



def extract_runtime_from_output(output_file):
    """
    Parses an output file to extract the total runtime of a process.

    This function scans through each line of a given output file, looking for specific markers
    that indicate the runtime of a process. It supports two formats:
    1. Lines containing 'real' which indicate time in 'minutesmseconds' format.
    2. Lines containing 'wall time' which indicate time split into hours, minutes, and seconds.

    The runtime is calculated and returned in seconds.

    Parameters:
    ----------
    output_file : str
        The path to the output file containing the runtime information.

    Returns:
    -------
    int
        The total runtime extracted from the file, in seconds.
    """
    # Initialize variables to store minutes and seconds
    minutes = seconds = 0

    # Open and read lines from the file
    with open(output_file, "r") as file:
        lines = file.readlines()

    # Parse each line to find runtime information
    for line in lines:
        if 'real' in line:
            words = line.split()
            # Extract minutes and seconds from the 'real' marker line
            minutes, seconds = map(int, words[1].split('m')[0].split('.'))
            
        elif 'wall time' in line:
            words = line.split(':')
            # Extract minutes and seconds from the 'wall time' marker line
            minutes = int(words[2])
            seconds = int(words[3])

    # Calculate total runtime in seconds
    runtime_seconds = (minutes * 60) + seconds

    return runtime_seconds


def generate_aggregate_stats(df_all, good_run=False):
    """
    Calculates a comprehensive set of statistical measures for each column in the provided DataFrame.

    Parameters:
    ----------
    df_all : pd.DataFrame
        The input DataFrame containing the data for which statistical measures are to be calculated.
    good_run : bool, optional
        A flag indicating whether to apply specific data filtering based on pre-defined criteria for "good runs". Default is False.

    Returns:
    -------
    tuple of lists:
        A tuple containing lists of various statistical measures (mean, standard deviation, quartiles, median,
        mean absolute deviation, count, kurtosis, skewness, and specific percentiles) for each column in df_all.
    """
    
    # Drop unused columns if present
    df_all.drop(columns=['timemsec', 'threadID'], errors='ignore', inplace=True)

    # Convert zero values to NaN to exclude them from calculations
    df_all.replace(0.0, np.NaN, inplace=True)

    # Initialize lists to store statistical measures
    statistics = {
        'means': [], 'stds': [], 'q1s': [], 'q3s': [], 'medians': [], 'mads': [],
        'counts': [], 'kurtosis': [], 'skews': [], 'q5s': [], 'q95s': [],
        'q01s': [], 'q99s': [], 'q10s': [], 'q90s': [], 'maxs': [], 'mins': []
    }

    # Define the percentiles to compute
    percentiles = {
        'q1s': 0.25, 'q3s': 0.75, 'q5s': 0.05, 'q95s': 0.95,
        'q01s': 0.01, 'q99s': 0.99, 'q10s': 0.10, 'q90s': 0.90
    }

    for field in df_all.columns:
        data = df_all[field]

        # Add condition-based filters for good runs if necessary
        if good_run:
            # Example filter: data = data[data <= 1.64] if field == "hbduration1" else data
            pass

        # Calculate and store each statistical measure
        statistics['means'].append(round(data.mean(), 3))
        statistics['stds'].append(data.std())
        statistics['medians'].append(data.median())
        statistics['mads'].append((np.abs(data - data.median())).median())
        statistics['counts'].append(data.count())
        statistics['kurtosis'].append(data.kurtosis())
        statistics['skews'].append(data.skew())
        statistics['maxs'].append(data.max())
        statistics['mins'].append(data.min())

        # Calculate and store specified percentiles
        for key, value in percentiles.items():
            statistics[key].append(data.quantile(value, interpolation='linear'))
    
    return tuple(statistics.values())

def extract_element_from_list(lst, index):
    """
    Extracts and returns a specific element from each sub-list within a list.

    This function iterates through a list of lists (or tuples) and extracts the element
    at a specified index from each sub-list or tuple. It is useful for extracting a
    particular field or value from a collection of similar items.

    Parameters:
    ----------
    lst : list of lists or tuples
        The input list containing sub-lists or tuples from which elements are to be extracted.
    index : int
        The index of the element to extract from each sub-list or tuple. Indices start at 0.

    Returns:
    -------
    list
        A list containing the elements extracted from the specified index of each sub-list or tuple.

    Example:
    --------
    >>> data = [['apple', 2], ['banana', 3], ['cherry', 5]]
    >>> extract_element_from_list(data, 0)
    ['apple', 'banana', 'cherry']

    >>> extract_element_from_list(data, 1)
    [2, 3, 5]
    """
    return [item[index] for item in lst]

def collect_root_paths(directory_paths):
    """
    Processes a list of directory paths, adjusting each path based on specific criteria and 
    compiling a unique list of root paths.

    This function iterates through each given path, moving up one directory level if the path
    includes specific keywords indicating it is a sub-directory of interest ('parameterfiles',
    'CoMD', or 'galaxy'). Each adjusted path is then added to a list of root paths, ensuring 
    no duplicates are included.

    Parameters:
    ----------
    directory_paths : list of str
        A list containing the directory paths to be processed.

    Returns:
    -------
    list of str
        A list of unique root paths derived from the original directory paths, adjusted according
        to the presence of specific keywords.

    Notes:
    -----
    - This function is particularly useful for scenarios where directory paths from various 
      simulations or parameter files need to be standardized to their root paths for further processing.
    - Assumes 'directory_paths' contains full paths to directories, not file paths.
    """

    rPaths = []

    for path in directory_paths:
        
        # Adjust the path if it contains specific keywords
        for keyword in ["parameterfiles", "CoMD", "galaxy"]:
            if keyword in path:
                path, _ = os.path.split(path)
                break  # Stop checking if any keyword is found

        # Add the adjusted path to the list if it's not already included
        if path not in rPaths:
            rPaths.append(path)

    return rPaths

def find_leaf_subdirectories(path):
    """
    Recursively finds and returns all leaf subdirectories within the given path.
    
    Parameters:
    - path: A string representing the directory path to search within.
    
    Returns:
    - A list of strings, where each string is a path to a leaf subdirectory.
    """
    leaf_subdirectories = []
    # Check if the current path is a directory and get all entries in it
    if os.path.isdir(path):
        entries = os.listdir(path)
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
        # If there are no subdirectories, then this is a leaf directory
        if not subdirectories:
            return [path]
        else:
            for subdir in subdirectories:
                # Recursively find leaf subdirectories in each subdirectory
                leaf_subdirectories.extend(find_leaf_subdirectories(os.path.join(path, subdir)))
    return leaf_subdirectories

# get problem sizes and resources (# threads), and wall times of comd 
# from the yaml file
# it returns problem size, # threads, and watl time of each run
def get_miniFE_quality_params(run_paths):
    """
    Extracts quality parameters from MiniFE benchmark runs located in specified directories.

    Parses through .yaml files within each run directory to collect information about the 
    problem size, number of processors, wall time, number of threads, total threads, and 
    nodes used in each MiniFE run.

    Parameters:
    ----------
    run_paths : list of str
        List of directory paths where MiniFE run data (.yaml files) are stored.

    Returns:
    -------
    tuple of lists:
        A tuple containing lists of extracted parameters: problem sizes, number of processors,
        wall times, number of threads, total threads, number of nodes, and processors per node.
    """
    # Define patterns for parameter extraction
    patterns = {
        "nx": "nx:",
        "ny": "ny:",
        "nz": "nz:",
        "threads": "OpenMP Max Threads::",
        "procs": "number of processors:",
        "steps": "Iterations: 200",
        "wall_time": "Total Program Time:"
    }

    # Initialize lists to store run parameters
    runs_prob_size, runs_nprocs, runs_wall_time = [], [], []
    runs_nthreads, runs_total_threads, runs_nnodes, procs_per_node = [], [], [], []

    # Process and clean the input paths
    run_paths = collect_root_paths(run_paths)

    for run in run_paths:
        files = [f for f in os.listdir(run) if f.endswith('yaml')]
        for file_name in files:
            file_path = os.path.join(run, file_name)
            with open(file_path, "r") as file:
                params = {}
                for line in file:
                    for key, pattern in patterns.items():
                        if re.search(pattern, line):
                            params[key] = int(line.split()[1])

                # Compute derived parameters
                if "nx" in params and "ny" in params and "nz" in params and "steps" in params:
                    prob_size = params["steps"] * params["nx"] * params["ny"] * params["nz"]
                    runs_prob_size.append(prob_size)

                if "procs" in params:
                    nnodes = params["procs"] / 16
                    runs_nnodes.append(nnodes)
                    runs_nprocs.append(params["procs"])
                    procs_per_node.append(16)

                if "threads" in params:
                    runs_nthreads.append(params["threads"])

                if "wall_time" in params:
                    runs_wall_time.append(params["wall_time"])

                if "procs" in params and "threads" in params:
                    total_threads = params["procs"] * params["threads"]
                    runs_total_threads.append(total_threads)

    return (runs_prob_size, runs_nprocs, runs_wall_time, runs_nthreads,
            runs_total_threads, runs_nnodes, procs_per_node)

def get_pennant_quality_params(run_paths):
    """
    Extracts quality parameters from PENNANT benchmark output files.

    Parses through output files within each run directory to collect information on MPI processes,
    threads, problem size, and run times. This function looks for specific markers indicating
    the number of nodes, MPI processes, threads, zones, cycles, and hydro cycle run times.

    Parameters:
    ----------
    run_paths : list of str
        List of directory paths where PENNANT run output files are stored.

    Returns:
    -------
    tuple of lists:
        A tuple containing lists of extracted parameters: problem sizes, number of MPI processes,
        wall times, number of threads, total threads, number of nodes, and processors per node.
    """
    # Define patterns for parameter extraction
    patterns = {
        "nodes": "Nodes (\d+)",
        "mpi": "MPI PE\s+=\s+(\d+)",
        "threads": "thread\s+=\s+(\d+)",
        "zones": "Zones:\s+(\d+)",
        "cycle": "cycle =\s+(\d+)",
        "hydro_time": "hydro cycle run time= (\d+\.\d+)"
    }

    # Initialize lists to store extracted parameters
    runs_prob_size, runs_nprocs, runs_wall_time = [], [], []
    runs_nthreads, runs_total_threads, runs_nnodes, procs_per_node = [], [], [], []

    # Clean the input paths
    run_paths = collect_root_paths(run_paths)

    for run in run_paths:
        files = [f for f in os.listdir(run) if f.endswith('out')]
        for file_name in files:
            file_path = os.path.join(run, file_name)
            with open(file_path, "r") as file:
                params = {}
                for line in file:
                    for key, pattern in patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            params[key] = int(match.group(1))

                # Compute and store derived parameters
                if all(k in params for k in ["nodes", "mpi", "threads", "zones", "cycle"]):
                    total_threads = params["mpi"] * params["threads"] * params["nodes"]
                    prob_size = params["zones"] * params["cycle"]
                    mpi_nodes = params["mpi"] / params["nodes"]

                    runs_wall_time.append(params.get("hydro_time", 0))
                    runs_nprocs.append(params["mpi"] * params["nodes"])
                    runs_nnodes.append(params["nodes"])
                    procs_per_node.append(params["mpi"])
                    runs_nthreads.append(params["threads"])
                    runs_total_threads.append(total_threads)
                    runs_prob_size.append(prob_size)

    return runs_prob_size, runs_nprocs, runs_wall_time, runs_nthreads, runs_total_threads, runs_nnodes, procs_per_node



def categorize_run_paths(run_paths):
    """
    Categorizes run paths based on specific keywords found in their paths.

    This function iterates through a list of directory paths, categorizing each run as either
    'good', 'anomalous', or 'misconfig' based on the presence of specific keywords within the path.
    The 'misconfig' category is commented out but can be included if conditions for it are defined.

    Parameters:
    ----------
    run_paths : list of str
        A list containing the directory paths to be categorized.

    Returns:
    -------
    list of str:
        A list of categories corresponding to each run path provided. Categories include
        'good', 'anomalous', and potentially 'misconfig'.

    Notes:
    -----
    - The function utilizes `collect_root_paths` to preprocess the input paths, potentially
      cleaning or standardizing them before categorization.
    - Currently, the 'misconfig' category is not automatically assigned but can be enabled
      by defining appropriate conditions within the loop.
    """
    # Preprocess and clean the input run paths
    run_paths = collect_root_paths(run_paths)

    # Initialize a list to store the categorization of each run path
    categories = []

    for run in run_paths:
        # Determine the category based on specific keywords in the run path
        if "APPEKG" in run:
            categories.append("good")
        elif "anomalousRuns" in run:
            categories.append("anomalous")
        elif "misConfig" in run:
            categories.append("misconfig")  # Uncomment and adjust as necessary

    return categories


def generate_aggregate_data(dir_path):
    """
    Aggregates data from all .csv files within a specified directory and its subdirectories
    into a single DataFrame. It also cleans the aggregated DataFrame by removing specific 
    columns if they exist.

    Parameters:
    ----------
    dir_path : str
        The path to the directory containing .csv files to aggregate.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing aggregated data from all .csv files found in the directory and
        its subdirectories. Columns 'timemsec' and 'threadID' are dropped if present.

    Notes:
    -----
    - This function walks through the directory and its subdirectories to find all .csv files.
    - It assumes that all .csv files have a consistent structure and can be concatenated.
    """
    aggregated_df = pd.DataFrame()

    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            if name.endswith(".csv"):
                file_path = os.path.join(root, name)
                temp_df = pd.read_csv(file_path)
                aggregated_df = pd.concat([aggregated_df, temp_df], ignore_index=True)

    # Drop specific columns if they exist in the DataFrame
    columns_to_drop = ['timemsec', 'threadID']
    aggregated_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    return aggregated_df

def get_model_stats(run_paths):
    """
    Aggregates statistics from model runs specified in the provided paths. Only runs
    containing 'APPEKG' in their path are processed.

    Parameters:
    ----------
    run_paths : list of str
        The list of run paths to process and generate statistics for.

    Returns:
    -------
    tuple:
        A tuple containing lists of various aggregated statistics for the runs,
        including means, standard deviations, quartiles, medians, mean absolute deviations (MADs),
        counts, kurtosis, skewness, and specified percentiles.
    """
    aggregated_df = pd.DataFrame()

    # Aggregate data from paths containing 'APPEKG'
    for run in run_paths:
        if 'APPEKG' in run:
            temp_df = generate_aggregate_data(run)
            aggregated_df = pd.concat([aggregated_df, temp_df], ignore_index=True)

    # Generate aggregate statistics from the combined DataFrame
    stats = generate_aggregate_stats(aggregated_df, good_run=True)

    return stats

def get_run_stats(run_paths):
    """
    Computes and aggregates run statistics for all specified run paths after preprocessing
    them to obtain root paths. It distinguishes between 'APPEKG' runs and others for statistical generation.

    Parameters:
    ----------
    run_paths : list of str
        The list of original run paths to process.

    Returns:
    -------
    tuple:
        A tuple containing lists of aggregated statistics across all processed runs.
    """
    # Preprocess paths to obtain root paths
    run_paths = collect_root_paths(run_paths)
    aggregated_stats = [[] for _ in range(11)]  # Initialize lists for each statistic

    for run in run_paths:
        df = generate_aggregate_data(run)
        is_good_run = 'APPEKG' in run
        stats = generate_aggregate_stats(df, good_run=is_good_run)

        # Append each statistic to its respective aggregated list
        for idx, stat in enumerate(stats):
            aggregated_stats[idx].append(stat)

    # Unpack aggregated statistics for return
    return tuple(aggregated_stats)

def get_lammps_quality_params(run_paths):
    """
    Extracts quality parameters from LAMMPS simulation output files.

    Parses through LAMMPS output files to gather information about the problem size, 
    number of processors, wall time, number of threads, total number of threads, 
    number of nodes, and processors per node.

    Parameters:
    ----------
    run_paths : list of str
        List of directory paths where LAMMPS run output files are stored.

    Returns:
    -------
    tuple:
        A tuple containing lists of extracted parameters: problem sizes, number of processors,
        wall times, number of threads, total threads, number of nodes, and processors per node.
    """
    # Initialize lists to store extracted parameters
    runs_prob_size, runs_nprocs, runs_wall_time = [], [], []
    runs_nthreads, runs_tot_nthreads, runs_nnodes, procs_per_node = [], [], [], []
    run_paths = find_leaf_subdirectories(run_paths)

    # Preprocess paths to obtain root paths
    run_paths = collect_root_paths(run_paths)
    classes = categorize_run_paths(run_paths) 
    # Define patterns for parameter extraction
    patterns = {
        "loop_time": re.compile(r"Loop time of\s+(\d+\.\d+)"),
        "cpu_use": re.compile(r"(\d+\.\d+)% CPU use with (\d+) MPI tasks x (\d+) OpenMP threads"),
        "total_wall_time": re.compile(r"Total wall time: (\d+):(\d{2}):(\d{2})")
    }
    for run in run_paths:
        files = [f for f in os.listdir(run) if f.endswith('out')]
        for file_name in files:
            file_path = os.path.join(run, file_name)
            with open(file_path, "r") as file:
                for line in file:
                    if match := patterns["loop_time"].search(line):
                        x = line.split()
                        nsteps = int(x[8])
                        atoms = int(x[11])
                        # nsteps, atoms = map(int, match.groups())
                        runs_prob_size.append(nsteps * atoms)
                    elif match := patterns["cpu_use"].search(line):
                        x = line.split()
                        nprocs = int(x[4]) 
                        nthreads = int(x[8])
                        # nprocs, nthreads = map(int, match.groups())
                        total_threads = nprocs * nthreads
                        runs_nprocs.append(nprocs)
                        runs_nthreads.append(nthreads)
                        runs_tot_nthreads.append(total_threads)
                        procs_per_node.append(16)  # Assuming a constant value of 16 procs per node
                        runs_nnodes.append(nprocs / 16)
                    elif match := patterns["total_wall_time"].search(line):
                        x = line.split()
                        wtime = x[3].split(":")
                        wall_time  = (int(wtime[0]) * 60 * 60) + (int(wtime[1]) * 60) + (int(wtime[2]))
                        runs_wall_time.append(wall_time)
    
    return runs_prob_size, runs_nprocs, runs_wall_time, runs_nthreads, runs_tot_nthreads, runs_nnodes, procs_per_node, classes    

def collect_comd_quality_params(run_paths):
    """
    Collects performance and configuration parameters from simulation run outputs.

    Parameters:
    - run_paths (list of str): List of directories containing the simulation output files.

    Returns:
    tuple: Contains lists of problem sizes, number of processors, wall times, number of threads,
           total threads, number of nodes, and processors per node for each run.
    """
    # Initialize lists to store the collected parameters
    runs_prob_size = []
    runs_nprocs = []
    runs_wall_time = []
    runs_nthreads = []
    runs_total_threads = []
    runs_nnodes = []
    procs_per_node = []

    # Patterns to identify relevant information in the output files
    pattern_timing_stats = "Timing Statistics Across"
    pattern_nsteps = "nSteps"
    pattern_total_atoms = "Total atoms        : "
    pattern_threading = "Threading: OpenMP"
    pattern_timer_total = "Timer: total"
    pattern_total = "Total:     "

    # Update run_paths with root paths
    run_paths = collect_root_paths(run_paths)

    for run_path in run_paths:
        for file_name in os.listdir(run_path):
            file_path = os.path.join(run_path, file_name)

            # Process output files
            if file_name.endswith('out'):
                with open(file_path, "r") as file:
                    nsteps, nblocks, procs, threads = 0, 0, 0, 0
                    for line in file:
                        if re.search(pattern_nsteps, line):
                            nsteps = int(line.split()[1])
                        elif re.search(pattern_total_atoms, line):
                            nblocks = int(line.split()[3])
                        elif re.search(pattern_timing_stats, line):
                            procs = int(line.split()[3])
                        elif re.search(pattern_threading, line):
                            threads = int(line.split()[2].split("(")[1])
                    
                    # Calculate and append problem size and thread information
                    prob_size = nsteps * nblocks
                    nprocs = threads * procs
                    runs_prob_size.append(prob_size)
                    runs_total_threads.append(nprocs)
                    runs_nprocs.append(procs)
                    runs_nthreads.append(threads)
                    procs_per_node.append(16)  # Assumes a fixed value for processors per node
                    runs_nnodes.append(procs / 16)

            # Process YAML files for wall time information
            elif file_name.endswith('yaml'):
                with open(file_path, "r") as file:
                    prev_line = False
                    for line in file:
                        if re.search(pattern_timer_total, line):
                            prev_line = True
                        elif re.search(pattern_total, line) and prev_line:
                            wall_time = float(line.split()[1])
                            runs_wall_time.append(wall_time)
                            break

    return (runs_prob_size, runs_nprocs, runs_wall_time, runs_nthreads, runs_total_threads, 
            runs_nnodes, procs_per_node)

def create_spreadsheet(df, file_name):
    """
    Creates an Excel spreadsheet from a pandas DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be exported to an Excel file.
    - file_name (str): The base name for the Excel file. The function appends
                       'jobQuality.xlsx' to this base name.

    Returns:
    None: The function saves an Excel file to the current working directory.
    """
    # Ensure the pandas library is available
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The provided data is not a pandas DataFrame.")
    
    # Append 'jobQuality.xlsx' to the provided file_name
    # full_file_name = f"{file_name}jobQuality.xlsx"
    
    # Export the DataFrame to an Excel file
    df.to_excel(file_name, index=False)

    print(f"Spreadsheet created successfully: {file_name}")

def extract_hb_stats(all_means, all_stds, all_q1s, all_q3s, all_medians, all_mads, all_counts, all_kur, all_skews, all_maxs, all_mins):
    """
    Extracts heart beat (HB) statistics from provided lists of lists, organizing them into individual lists.

    Parameters:
    - all_means, all_stds, all_q1s, all_q3s, all_medians, all_mads, all_counts, all_kur, all_skews, all_maxs, all_mins
      (list of lists): Lists containing statistical metrics across multiple datasets.

    Returns:
    tuple of lists: Each list contains a specific statistical metric across datasets.
    """
    # Utilize a more efficient approach by iterating once and extracting all metrics simultaneously
    num_datasets = len(all_means[0])
    hb_data = [extract_element_from_list(all_means, i) for i in range(num_datasets)]
    hb_stds = [extract_element_from_list(all_stds, i) for i in range(num_datasets)]
    hb_q1s = [extract_element_from_list(all_q1s, i) for i in range(num_datasets)]
    hb_q3s = [extract_element_from_list(all_q3s, i) for i in range(num_datasets)]
    medians = [extract_element_from_list(all_medians, i) for i in range(num_datasets)]
    mads = [extract_element_from_list(all_mads, i) for i in range(num_datasets)]
    counts = [extract_element_from_list(all_counts, i) for i in range(num_datasets)]
    kurs = [extract_element_from_list(all_kur, i) for i in range(num_datasets)]
    skews = [extract_element_from_list(all_skews, i) for i in range(num_datasets)]
    maxs = [extract_element_from_list(all_maxs, i) for i in range(num_datasets)]
    mins = [extract_element_from_list(all_mins, i) for i in range(num_datasets)]

    return hb_data, hb_stds, hb_q1s, hb_q3s, medians, mads, counts, kurs, skews, maxs, mins

def create_dataframe(prob_size, wall_times, runs_nprocs, qualities, classes, runs_nthreads, runs_total_threads, runs_nnodes, runs_procs_per_node):
    """
    Creates a pandas DataFrame from given lists, organizing computational job or run attributes.

    Parameters:
    - prob_size (list): List of problem sizes for each run.
    - wall_times (list): List of wall times for each run.
    - runs_nprocs (list): List of the number of processes for each run.
    - qualities (list): List of quality metrics for each run.
    - classes (list): List of classification labels for each run.
    - runs_nthreads (list): List of the number of threads per process for each run.
    - runs_total_threads (list): Total number of threads for each run.
    - runs_nnodes (list): List of the number of nodes used for each run.
    - runs_procs_per_node (list): List of the number of processes per node for each run.

    Returns:
    pandas.DataFrame: DataFrame with each parameter represented as a column.
    """

    # Create the DataFrame
    df = pd.DataFrame({
        '# Nodes': runs_nnodes,
        '# Procs/Node': runs_procs_per_node,
        'Total Processes': runs_nprocs,
        '# Threads/Process': runs_nthreads,
        'Total # Threads': runs_total_threads,
        'Wall Time': wall_times,
        'Problem Size': prob_size,
        'Qualities': qualities,
        'Classes': classes,
    })

    return df

def append_hb_data_to_df(df, hb_data, hb_stds, hb_q1s, hb_q3s, medians, mads, counts, kurs, skews, maxs, mins):
    """
    Appends descriptive statistics for heartbeat (HB) data to an existing DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to append data to.
    - hb_data, hb_stds, hb_q1s, hb_q3s, medians, mads, counts, kurs, skews, maxs, mins (list):
      Lists containing the statistical measures for heartbeat data. Each list should have an even
      number of elements, representing pairs of measurements.

    Returns:
    - pandas.DataFrame: The original DataFrame with appended heartbeat statistics.
    """
    for j in range(len(hb_data) // 2):
        # Define base column names for the two related sets of statistics
        hb_name1 = f"hb{j+1}c"
        hb_name2 = f"hb{j+1}d"

        # Append statistics for the first set
        df[f"{hb_name1} mean"] = hb_data[j*2]
        df[f"{hb_name1} Std"] = hb_stds[j*2]
        df[f"{hb_name1} Q1"] = hb_q1s[j*2]
        df[f"{hb_name1} Q3"] = hb_q3s[j*2]
        df[f"{hb_name1} Median"] = medians[j*2]
        df[f"{hb_name1} MAD"] = mads[j*2]
        df[f"{hb_name1} # data points"] = counts[j*2]
        df[f"{hb_name1} Kurtosis"] = kurs[j*2]
        df[f"{hb_name1} Skew"] = skews[j*2]
        df[f"{hb_name1} Min"] = mins[j*2]
        df[f"{hb_name1} Max"] = maxs[j*2]

        # Append statistics for the second set
        df[f"{hb_name2} mean"] = hb_data[j*2+1]
        df[f"{hb_name2} Std"] = hb_stds[j*2+1]
        df[f"{hb_name2} Q1"] = hb_q1s[j*2+1]
        df[f"{hb_name2} Q3"] = hb_q3s[j*2+1]
        df[f"{hb_name2} Median"] = medians[j*2+1]
        df[f"{hb_name2} MAD"] = mads[j*2+1]
        df[f"{hb_name2} # data points"] = counts[j*2+1]
        df[f"{hb_name2} Kurtosis"] = kurs[j*2+1]
        df[f"{hb_name2} Skew"] = skews[j*2+1]
        df[f"{hb_name2} Min"] = mins[j*2+1]
        df[f"{hb_name2} Max"] = maxs[j*2+1]

    return df

def get_mini_amr_quality_params(run_paths):
    """
    Extracts problem sizes, the number of processes, wall times, the number of threads, 
    total threads, the number of nodes, and processes per node from the MiniAMR output files.

    Parameters:
    - run_paths (list of str): List of directories containing the output files.

    Returns:
    tuple: Contains lists of problem sizes, number of processes, wall times, number of threads,
           total threads, number of nodes, and processes per node for each run.
    """
    pattern_timesteps = "Number of timesteps is"
    pattern_blocks = "Total blocks           :"
    pattern_summary = "Summary: "
    
    runs_prob_size = []
    runs_nprocs = []
    runs_wall_time = []
    runs_nthreads = []
    runs_total_threads = []
    runs_nnodes = []
    procs_per_node = []

    run_paths = collect_root_paths(run_paths)

    for run in run_paths:
        for file_name in os.listdir(run):
            if file_name.endswith('out'):
                file_path = os.path.join(run, file_name)
                with open(file_path, "r") as file:
                    for line in file:
                        if re.search(pattern_timesteps, line):
                            nsteps = int(line.split()[4])
                        if re.search(pattern_blocks, line):
                            nblocks = int(line.split()[3])
                        if re.search(pattern_summary, line):
                            summary_data = line.split()
                            procs = int(summary_data[2])
                            threads = int(summary_data[4])
                            wall_time = float(summary_data[8])
                            nprocs = procs * threads

                            runs_total_threads.append(nprocs)
                            runs_wall_time.append(wall_time)
                            procs_per_node.append(8)  # Assumes a fixed value for procs per node
                            runs_nthreads.append(threads)
                            runs_nnodes.append(procs / 8)  # Assumes a fixed value for procs per node
                            runs_nprocs.append(procs)
                            
                            prob_size = nsteps * nblocks
                            runs_prob_size.append(prob_size)

    return runs_prob_size, runs_nprocs, runs_wall_time, runs_nthreads, runs_total_threads, runs_nnodes, procs_per_node

def get_gadget_quality_params(run_paths):
    """
    Extracts quality parameters from Gadget simulation output files, including the total number of particles,
    the number of processors, wall times, and the computation resources utilized per run.

    Parameters:
    - run_paths (list of str): List of directories containing the Gadget simulation output files.

    Returns:
    tuple: Contains lists of problem sizes (number of particles), number of processors, wall times, number of threads,
           total threads, number of nodes, and processors per node for each simulation run.
    """
    pattern_particles = "Total number of particles"
    pattern_procs = "Running on "
    runs_prob_size = []
    runs_nprocs = []
    runs_wall_time = []
    runs_nthreads = []
    runs_total_threads = []
    runs_nnodes = []
    procs_per_node = []

    run_paths = collect_root_paths(run_paths)

    for run in run_paths:
        for file_name in os.listdir(run):
            file_path = os.path.join(run, file_name)

            if file_name.endswith('out'):
                with open(file_path, "r") as file:
                    for line in file:
                        if re.search(pattern_particles, line):
                            nparticles = int(line.split()[5])
                            runs_prob_size.append(nparticles)
                            runs_nthreads.append(1)  # Assuming 1 thread per process
                        elif re.search(pattern_procs, line):
                            nprocs = int(line.split()[2])
                            runs_nprocs.append(nprocs)
                            procs_per_node.append(32)  # Assuming a fixed value for procs per node
                            runs_total_threads.append(nprocs)
                            runs_nnodes.append(nprocs / 32)  # Assuming a fixed value for procs per node

            elif file_name.endswith('error'):
                wall_time = extract_runtime_from_output(file_path)
                runs_wall_time.append(wall_time)
    
    return runs_prob_size, runs_nprocs, runs_wall_time, runs_nthreads, runs_total_threads, runs_nnodes, procs_per_node



def compute_quality(problem_size, run_times, resources):
    """
    Computes the quality of each run based on problem size, run times, and resources used.

    The quality for each run is defined as the problem size divided by the product of resources used and the run time.

    Parameters:
    - problem_size (list of int/float): A list containing the problem sizes for each run.
    - run_times (list of int/float): A list containing the run times for each run.
    - resources (list of int): A list containing the amount of resources used for each run.

    Returns:
    - list of float: A list containing the computed quality for each run, rounded to 2 decimal places.
    """
    qualities = []

    for size, time, resource in zip(problem_size, run_times, resources):
        if resource * time == 0:
            raise ValueError("Resource times time cannot be zero.")
        quality = size / (resource * time)
        qualities.append(round(quality, 2))

    return qualities

def get_run_statistics(run_paths):
    """
    Collects and aggregates statistical data from a series of runs located in subdirectories.
    
    This function processes each run by aggregating data and computing statistics such as mean,
    standard deviation, quartiles, median, median absolute deviation (MAD), count, kurtosis, skewness,
    and minimum and maximum values.
    
    Args:
        run_paths (list): A list of directory paths that contain run data.
    
    Returns:
        tuple of lists: Returns multiple lists containing the aggregated statistics for all runs:
                        means, standard deviations, first quartiles, third quartiles, medians, MADs,
                        counts, kurtosis values, skewness values, maximums, and minimums.
    """
    
    # Expand run_paths to include all subdirectories
    run_paths = find_leaf_subdirectories(run_paths)

    # Preprocess paths to obtain root paths
    run_paths = collect_root_paths(run_paths)
    
    
    # Initialize lists to store aggregated statistics for all runs
    all_means, all_stds, all_q1s, all_q3s, all_medians, all_mads, all_counts, all_kur, all_skews, all_maxs, all_mins = ([] for _ in range(11))
    
    # Iterate through each run, aggregating and computing statistics
    for run in run_paths:
        # Aggregate data for the current run
        df = generate_aggregate_data(run)
        
        # Compute statistics for the current data frame
        means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews, _, _, _, _, _, _, maxs, mins = generate_aggregate_stats(df)
        
        # Append the computed statistics to their respective lists
        all_means.append(means)
        all_stds.append(stds)
        all_q1s.append(q1s)
        all_q3s.append(q3s)
        all_medians.append(medians)
        all_mads.append(mads)
        all_counts.append(counts)
        all_kur.append(kurtosis)
        all_skews.append(skews)
        all_maxs.append(maxs)
        all_mins.append(mins)
    
    # Return the aggregated statistics
    return all_means, all_stds, all_q1s, all_q3s, all_medians, all_mads, all_counts, all_kur, all_skews, all_maxs, all_mins
def generate_aggregate_data(dir_path):
    """
    Aggregates data from CSV files found in a directory and its subdirectories into a single DataFrame.
    
    This function searches for CSV files within the specified directory path, including all subdirectories,
    and concatenates their contents into a single pandas DataFrame. Columns specific to time in milliseconds
    ('timemsec') and thread IDs ('threadID') are dropped if present, as they are presumably not needed for
    the aggregate analysis.
    
    Args:
        dir_path (str): The path to the directory containing the CSV files to aggregate.
    
    Returns:
        pd.DataFrame: A pandas DataFrame containing the aggregated data from all found CSV files,
                      with specific columns removed as noted.
    """
    # Initialize an empty DataFrame to store the aggregated data
    df = pd.DataFrame()
    
    # Walk through the directory structure, aggregating CSV files
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            if name.endswith(".csv"):
                # Construct the full path to the CSV file
                file_path = os.path.join(root, name)
                # Read the CSV file into a temporary DataFrame
                temp_df = pd.read_csv(file_path)
                # Concatenate the temporary DataFrame with the aggregated DataFrame
                df = pd.concat([df, temp_df], ignore_index=True)
    
    # Drop 'timemsec' and 'threadID' columns if they exist
    df.drop(columns=['timemsec', 'threadID'], errors='ignore', inplace=True)
    
    return df

def extract_hb_stats(all_means, all_stds, all_q1s, all_q3s, all_medians, all_mads, all_counts, all_kur, all_skews, all_maxs, all_mins):
    """
    Extracts heart beat (HB) statistics from aggregated lists of data.
    
    Given lists of aggregated statistical measures (means, standard deviations, quartiles, etc.), this function
    iterates over the indices of these lists to extract and compile the respective statistics for each index
    across all provided measures. This is primarily used for analyzing heart beat data aggregated from multiple
    sources or runs.
    
    Args:
        all_means (list): List of mean values.
        all_stds (list): List of standard deviation values.
        all_q1s (list): List of first quartile values.
        all_q3s (list): List of third quartile values.
        all_medians (list): List of median values.
        all_mads (list): List of median absolute deviation values.
        all_counts (list): List of count values.
        all_kur (list): List of kurtosis values.
        all_skews (list): List of skewness values.
        all_maxs (list): List of maximum values.
        all_mins (list): List of minimum values.
    
    Returns:
        tuple: A tuple containing lists of the extracted statistics for heart beat data across all indices.
    """
    
    # Initialize lists to store extracted statistics
    hb_data, hb_stds, hb_q1s, hb_q3s, medians, mads, counts, kurs, skews, maxs, mins = ([] for _ in range(11))
    
    # The loop iterates based on the length of the first list of means, assuming all lists are of equal length
    for i in range(len(all_means[0])):
        hb_data.append(extract(all_means, i))
        hb_stds.append(extract(all_stds, i))
        hb_q1s.append(extract(all_q1s, i))
        hb_q3s.append(extract(all_q3s, i))
        medians.append(extract(all_medians, i))
        mads.append(extract(all_mads, i))
        counts.append(extract(all_counts, i))
        kurs.append(extract(all_kur, i))
        skews.append(extract(all_skews, i))
        maxs.append(extract(all_maxs, i))
        mins.append(extract(all_mins, i))
    
    return hb_data, hb_stds, hb_q1s, hb_q3s, medians, mads, counts, kurs, skews, maxs, mins

def extract(data_list, index):
    """
    Extracts and aggregates data for a given index across all lists of a specific statistic.
    
    This helper function is used to abstract the process of iterating through lists of data
    and extracting the value at a given index.
    
    Args:
        data_list (list of lists): The aggregated list of data from which to extract.
        index (int): The index for which data is to be extracted across all lists.
    
    Returns:
        list: A list of extracted data for the given index across all input lists.
    """
    return [data[index] for data in data_list]

def append_hb_data_to_df(df, hb_data, hb_stds, hb_q1s, hb_q3s, medians, mads, counts, kurs, skews, maxs, mins):
    """
    Appends heart beat (HB) statistical data to the provided DataFrame.
    
    This function iterates over provided HB statistical data, appending it to the DataFrame in a structured manner.
    Each statistic is appended as a new column with a naming convention that includes the statistic name and its
    sequence number, differentiating between 'c' and 'd' types for each statistic pair.
    
    Args:
        df (pd.DataFrame): The DataFrame to which the HB statistical data will be appended.
        hb_data (list): List of mean values for HB data.
        hb_stds (list): List of standard deviation values for HB data.
        hb_q1s (list): List of first quartile values for HB data.
        hb_q3s (list): List of third quartile values for HB data.
        medians (list): List of median values for HB data.
        mads (list): List of median absolute deviation values for HB data.
        counts (list): List of data point counts for HB data.
        kurs (list): List of kurtosis values for HB data.
        skews (list): List of skewness values for HB data.
        maxs (list): List of maximum values for HB data.
        mins (list): List of minimum values for HB data.
    
    Returns:
        pd.DataFrame: The updated DataFrame with appended HB statistical data.
    """
    
    for j in range(len(hb_data) // 2):
        # Define column name prefixes for 'c' and 'd' types
        hbname1 = f"hb{j+1}c"
        hbname2 = f"hb{j+1}d"
        
        # Append 'c' type statistical data
        df[f"{hbname1} mean"] = hb_data[j*2]
        df[f"{hbname1} Std"] = hb_stds[j*2]
        df[f"{hbname1} Q1"] = hb_q1s[j*2]
        df[f"{hbname1} Q3"] = hb_q3s[j*2]
        df[f"{hbname1} Median"] = medians[j*2]
        df[f"{hbname1} MAD"] = mads[j*2]
        df[f"{hbname1} # data points"] = counts[j*2]
        df[f"{hbname1} Kurtosis"] = kurs[j*2]
        df[f"{hbname1} Skew"] = skews[j*2]
        df[f"{hbname1} Min"] = mins[j*2]
        df[f"{hbname1} Max"] = maxs[j*2]
        
        # Append 'd' type statistical data
        df[f"{hbname2} mean"] = hb_data[j*2+1]
        df[f"{hbname2} Std"] = hb_stds[j*2+1]
        df[f"{hbname2} Q1"] = hb_q1s[j*2+1]
        df[f"{hbname2} Q3"] = hb_q3s[j*2+1]
        df[f"{hbname2} Median"] = medians[j*2+1]
        df[f"{hbname2} MAD"] = mads[j*2+1]
        df[f"{hbname2} # data points"] = counts[j*2+1]
        df[f"{hbname2} Kurtosis"] = kurs[j*2+1]
        df[f"{hbname2} Skew"] = skews[j*2+1]
        df[f"{hbname2} Min"] = mins[j*2+1]
        df[f"{hbname2} Max"] = maxs[j*2+1]
        
    # Return the updated DataFrame
    return pd.DataFrame(df)

def get_model_stats(paths):
    """
    Aggregates data from runs containing 'APPEKG' in their paths and computes various statistics.
    
    This function filters runs that include 'APPEKG' in their path, aggregates their data into a single
    DataFrame, and then calculates various statistical measures, including means, standard deviations,
    quartiles, medians, median absolute deviations (MADs), counts, kurtosis, skewness, and specific
    percentiles.
    
    Args:
        paths (list): A list of directory paths that contain run data.
    
    Returns:
        tuple: A tuple containing the calculated statistical measures in the following order:
               means, standard deviations, first quartiles (Q1s), third quartiles (Q3s), medians,
               MADs, counts, kurtosis, skewness, 5th percentiles (q5s), 95th percentiles (q95s),
               1st percentiles (q01s), 99th percentiles (q99s), 10th percentiles (q10s),
               90th percentiles (q90s), maximums, and minimums.
    """
    # Initialize an empty DataFrame for aggregating data
    aggregated_df = pd.DataFrame()
    
    # Iterate through each provided path to filter and aggregate data
    for run in paths:
        # Check if 'APPEKG' is in the run's path
        if 'APPEKG' in run:
            # Generate aggregated data for the current run
            _df = generate_aggregate_data(run)
            # Concatenate the current run's data with the aggregated DataFrame
            aggregated_df = pd.concat([aggregated_df, _df], ignore_index=True)
    
    # Generate statistical measures from the aggregated data
    stats = generate_aggregate_stats(aggregated_df, True)
    
    # Unpack and return the statistical measures
    means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews, q5s, q95s, q01s, q99s, q10s, q90s, maxs, mins = stats
    return means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews, q5s, q95s, q01s, q99s, q10s, q90s, maxs, mins

def parse_arguments():
    """
    Parses command-line arguments, expecting an application name and a path to run data.
    """
    parser = argparse.ArgumentParser(description='Compute Quality Metrics for Computational Jobs')
    parser.add_argument('-a', '--application', type=str, help="Specify application name")
    parser.add_argument('-p', '--path', type=str, required=True, help="Specify the path of the runs")
    return parser.parse_args()

def main():
    """
    Main execution function. It reads command-line arguments to determine the application
    and the path to run data. Based on the application specified, it calls the appropriate
    function to compute quality metrics and generate an Excel spreadsheet.
    """
    args = parse_arguments()

    # Mapping of application names to their respective functions
    app_functions = {
        'lammps': get_lammps_quality_params,
        'minife': get_miniFE_quality_params,
        'miniamr': get_mini_amr_quality_params,
        'comd': collect_comd_quality_params,
        'gadget': get_gadget_quality_params,
        'pennant': get_pennant_quality_params
    }

    # Check if the specified application is supported
    if args.application in app_functions:
        # Retrieve quality parameters based on the application
        runs_prob_size, runs_nprocs, runs_wall_time, runs_nthreads, runs_tot_nthreads, \
            runs_nnodes, procs_per_node, classes = app_functions[args.application](args.path)
        
        
        # Compute quality metrics
        qualities = compute_quality(runs_prob_size, runs_wall_time, runs_tot_nthreads )

        # Organize data into a DataFrame
        df = create_dataframe(runs_prob_size,runs_wall_time, runs_nprocs, qualities, classes, runs_nthreads, runs_tot_nthreads, \
            runs_nnodes, procs_per_node)  # Assuming create_dataframe can handle the data structure
        # Generate a spreadsheet
        spreadsheet_filename = f"{args.application}_stats_qualities.xlsx"
        spreadsheet_filename2 = f"{args.application}_stats_model.xlsx"
        # create_spreadsheet(df, spreadsheet_filename)
        # print(f"Spreadsheet created: {spreadsheet_filename}")
        
        allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts, allKur, allSkwes, \
            allMaxs, allMins = get_run_statistics(args.path)
        hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins = extract_hb_stats(allMeans,\
            allStds, allQ1s, allQ3s, allMedians, allmads, allCounts,allKur, allSkwes, allMaxs, allMins)
        df2 = append_hb_data_to_df(df, hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, \
            kurs, skews, maxs, mins)
        create_spreadsheet(df2, spreadsheet_filename )
        # Expand run_paths to include all subdirectories
        run_paths = find_leaf_subdirectories(args.path)

        # Preprocess paths to obtain root paths
        run_paths = collect_root_paths(run_paths)
    
        means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews,\
            q5s, q95s, q01s, q99s, q10s, q90s, maxs, mins = get_model_stats(run_paths)
        hbnames = []
        for i in range(int(len(means)/2)):
            hbname = "hb" + str(i+1) + "-count" 
            hbnames.append(hbname)
            hbname = "hb" + str(i+1) + "-duration"
            hbnames.append(hbname)
        data_dict = {'metric': hbnames, 'Mean': means, 'std-dev': stds, 'Q1': q1s, 'Q3': q3s, 'Median': medians,\
            'MAD': mads, '# data points': counts, 'kurtosis': kurtosis, 'skewness': skews,\
                'max' :maxs, 'min' :mins, 'q5s' :q5s, 'q95s' :q95s, 'q01s':q01s, 'q99s':q99s, 'q10s':q10s, 'q90s': q90s }
        df3 = pd.DataFrame(data_dict)
        # Specify the filename
        create_spreadsheet(df3, spreadsheet_filename2 )
        
    
    else:
        print(f"Application '{args.application}' is not currently supported.")

if __name__ == "__main__":
    main()
# # main
# args = parser.parse_args()
# app = args.application
# path = args.path

# if app == None or app == 'lammps':
#     probSize, runsNProcs, wallTimes, runsNThreads, runsTotNThreads, runsNnodes, runsProcsPerNode  =  get_lammps_quality_params(path)
#     classes = categorize_run_paths(path)
#     qualities = compute_quality(probSize, wallTimes, runsTotNThreads)
#     df = create_dataframe(probSize, wallTimes, runsNProcs, qualities, classes, runsNThreads, runsTotNThreads, runsNnodes, runsProcsPerNode)
#     allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts, allKur, allSkwes, allMaxs, allMins = get_run_stats(path)
#     hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins = extract_hb_stats(allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts,allKur, allSkwes, allMaxs, allMins)
#     df2 = append_hb_data_to_df(df, hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins)
#     create_spreadsheet(df2, app )
#     runPaths = collect_root_paths(path)
    
#     means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews, q5s, q95s, q01s, q99s, q10s, q90s, maxs, mins = get_model_stats(runPaths)
#     hbnames = []
#     for i in range(int(len(means)/2)):
#         hbname = "hb" + str(i+1) + "-count" 
#         hbnames.append(hbname)
#         hbname = "hb" + str(i+1) + "-duration"
#         hbnames.append(hbname)
#     data_dict = {'metric': hbnames, 'Mean': means, 'std-dev': stds, 'Q1': q1s, 'Q3': q3s, 'Median': medians, 'MAD': mads, '# data points': counts, 'kurtosis': kurtosis, 'skewness': skews, 'max' :maxs, 'min' :mins, 'q5s' :q5s, 'q95s' :q95s, 'q01s':q01s, 'q99s':q99s, 'q10s':q10s, 'q90s': q90s }
#     df = pd.DataFrame(data_dict)
#     # Specify the filename
#     filename = 'LAMMPSModel.xlsx'
#     df.to_excel(filename, index=False)

# elif app == 'minife':  
#     probSize, runsNProcs, wallTimes, runsNThreads, runsTotNThreads, runsNnodes, runsProcsPerNode  =  get_miniFE_quality_params(path)
#     classes = categorize_run_paths(path)
#     qualities = compute_quality(probSize, wallTimes, runsTotNThreads)
#     df = create_dataframe(probSize, wallTimes, runsNProcs, qualities, classes, runsNThreads, runsTotNThreads, runsNnodes, runsProcsPerNode)
#     allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts, allKur, allSkwes, allMaxs, allMins = get_run_stats(path)
#     hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins = extract_hb_stats(allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts,allKur, allSkwes, allMaxs, allMins)
#     df2 = append_hb_data_to_df(df, hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins)
#     create_spreadsheet(df2, app )
#     runPaths = collect_root_paths(path)
    
#     means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews, q5s, q95s, q01s, q99s, q10s, q90s, maxs, mins = get_model_stats(runPaths)
#     hbnames = []
#     for i in range(int(len(means)/2)):
#         hbname = "hb" + str(i+1) + "-count" 
#         hbnames.append(hbname)
#         hbname = "hb" + str(i+1) + "-duration"
#         hbnames.append(hbname)
#     data_dict = {'metric': hbnames, 'Mean': means, 'std-dev': stds, 'Q1': q1s, 'Q3': q3s, 'Median': medians, 'MAD': mads, '# data points': counts, 'kurtosis': kurtosis, 'skewness': skews, 'max' :maxs, 'min' :mins, 'q5s' :q5s, 'q95s' :q95s, 'q01s':q01s, 'q99s':q99s, 'q10s':q10s, 'q90s': q90s }
#     df = pd.DataFrame(data_dict)
#     # Specify the filename
#     filename = 'miniFEModel.xlsx'
#     df.to_excel(filename, index=False)
    
# elif app == 'miniamr' or app == 'miniAMR':
#     runsProbSize, runsNprocs, runsWallTime, runsNThreads, runsTotalThreads, runsNnodes, procsPerNode =  get_mini_amr_quality_params(path)
#     classes = categorize_run_paths(path)
#     qualities = compute_quality(runsProbSize, runsWallTime, runsTotalThreads)
#     df = create_dataframe(runsProbSize, runsWallTime, runsNprocs, qualities, classes, runsNThreads, runsTotalThreads, runsNnodes, procsPerNode)
#     allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts, allKur, allSkwes, allMaxs, allMins = get_run_stats(path)
#     hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins = extract_hb_stats(allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts,allKur, allSkwes, allMaxs, allMins)
#     df2 = append_hb_data_to_df(df, hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins)
#     create_spreadsheet(df2, app )
#     runPaths = collect_root_paths(path)
    
#     means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews, q5s, q95s, q01s, q99s, q10s, q90s, maxs, mins = get_model_stats(runPaths)
#     hbnames = []
#     for i in range(int(len(means)/2)):
#         hbname = "hb" + str(i+1) + "-count" 
#         hbnames.append(hbname)
#         hbname = "hb" + str(i+1) + "-duration"
#         hbnames.append(hbname)
#     data_dict = {'metric': hbnames, 'Mean': means, 'std-dev': stds, 'Q1': q1s, 'Q3': q3s, 'Median': medians, 'MAD': mads, '# data points': counts, 'kurtosis': kurtosis, 'skewness': skews, 'max' :maxs, 'min' :mins, 'q5s' :q5s, 'q95s' :q95s, 'q01s':q01s, 'q99s':q99s, 'q10s':q10s, 'q90s': q90s }
#     df = pd.DataFrame(data_dict)
#     # Specify the filename
#     filename = 'miniamrModel.xlsx'
#     df.to_excel(filename, index=False)

# elif app == 'comd' or app == 'CoMD':
#     runsProbSize, runsNprocs, runsWallTime, runsNThreads, runsTotalThreads, runsNnodes, procsPerNode =  collect_comd_quality_params(path)
#     classes = categorize_run_paths(path)
#     qualities = compute_quality(runsProbSize, runsWallTime, runsTotalThreads)
#     print(len(runsProbSize))
#     print(len(classes))
#     print(len(qualities))
#     df = create_dataframe(runsProbSize, runsWallTime, runsNprocs, qualities, classes, runsNThreads, runsTotalThreads, runsNnodes, procsPerNode)
#     allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts, allKur, allSkwes, allMaxs, allMins = get_run_stats(path)
#     hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins = extract_hb_stats(allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts,allKur, allSkwes, allMaxs, allMins)
#     df2 = append_hb_data_to_df(df, hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins)
#     create_spreadsheet(df2, app )
#     runPaths = collect_root_paths(path)
    
#     means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews, q5s, q95s, q01s, q99s, q10s, q90s, maxs, mins = get_model_stats(runPaths)
#     hbnames = []
#     for i in range(int(len(means)/2)):
#         hbname = "hb" + str(i+1) + "-count" 
#         hbnames.append(hbname)
#         hbname = "hb" + str(i+1) + "-duration"
#         hbnames.append(hbname)
#     data_dict = {'metric': hbnames, 'Mean': means, 'std-dev': stds, 'Q1': q1s, 'Q3': q3s, 'Median': medians, 'MAD': mads, '# data points': counts, 'kurtosis': kurtosis, 'skewness': skews, 'max' :maxs, 'min' :mins, 'q5s' :q5s, 'q95s' :q95s, 'q01s':q01s, 'q99s':q99s, 'q10s':q10s, 'q90s': q90s }
#     df = pd.DataFrame(data_dict)
#     # Specify the filename
#     filename = 'comdModel.xlsx'
#     df.to_excel(filename, index=False)

# elif app == 'gadget' or app == 'Gadget':
#     runsProbSize, runsNprocs, runsWallTime, runsNThreads, runsTotalThreads, runsNnodes, procsPerNode =  getGadgetQualityParams(path)
#     classes = categorize_run_paths(path)
#     qualities = compute_quality(runsProbSize, runsWallTime, runsTotalThreads)
#     df = create_dataframe(runsProbSize, runsWallTime, runsNprocs, qualities, classes, runsNThreads, runsTotalThreads, runsNnodes, procsPerNode)
#     allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts, allKur, allSkwes, allMaxs, allMins = get_run_stats(path)
#     hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins = extract_hb_stats(allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts,allKur, allSkwes, allMaxs, allMins)
#     df2 = append_hb_data_to_df(df, hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins)
#     create_spreadsheet(df2, app )
#     print(df2)
#     runPaths = collect_root_paths(path)


    
#     means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews, q5s, q95s, q01s, q99s, q10s, q90s, maxs, mins = get_model_stats(runPaths)
#     hbnames = []
#     for i in range(int(len(means)/2)):
#         hbname = "hb" + str(i+1) + "-count" 
#         hbnames.append(hbname)
#         hbname = "hb" + str(i+1) + "-duration"
#         hbnames.append(hbname)
#     data_dict = {'metric': hbnames, 'Mean': means, 'std-dev': stds, 'Q1': q1s, 'Q3': q3s, 'Median': medians, 'MAD': mads, '# data points': counts, 'kurtosis': kurtosis, 'skewness': skews, 'max' :maxs, 'min' :mins, 'q5s' :q5s, 'q95s' :q95s, 'q01s':q01s, 'q99s':q99s, 'q10s':q10s, 'q90s': q90s }
#     df = pd.DataFrame(data_dict)
#     # Specify the filename
#     filename = 'gadgetModel.xlsx'
#     df.to_excel(filename, index=False)

# elif app == 'pennant' or app == 'PENNANT':
#     runsProbSize, runsNprocs, runsWallTime, runsNThreads, runsTotalThreads, runsNnodes, procsPerNode   =  getPENNANTQualityParams(path)
#     qualities = compute_quality(runsProbSize, runsWallTime, runsTotalThreads)
#     classes = categorize_run_paths(path)
#     df = create_dataframe(runsProbSize, runsWallTime, runsNprocs, qualities, classes, runsNThreads, runsTotalThreads, runsNnodes, procsPerNode)
    
#     allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts, allKur, allSkwes, allMaxs, allMins = get_run_stats(path)
#     hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins = extract_hb_stats(allMeans, allStds, allQ1s, allQ3s, allMedians, allmads, allCounts,allKur, allSkwes, allMaxs, allMins)
#     df2 = append_hb_data_to_df(df, hbData, hbStds, hbq1s, hbq3s, medians, mads, counts, kurs, skews, maxs, mins)
#     create_spreadsheet(df2, app )
#     runPaths = collect_root_paths(path)
    
#     means, stds, q1s, q3s, medians, mads, counts, kurtosis, skews, q5s, q95s, q01s, q99s, q10s, q90s, maxs, mins = get_model_stats(runPaths)
#     hbnames = []
#     for i in range(int(len(means)/2)):
#         hbname = "hb" + str(i+1) + "-count" 
#         hbnames.append(hbname)
#         hbname = "hb" + str(i+1) + "-duration"
#         hbnames.append(hbname)
#     data_dict = {'metric': hbnames, 'Mean': means, 'std-dev': stds, 'Q1': q1s, 'Q3': q3s, 'Median': medians, 'MAD': mads, '# data points': counts, 'kurtosis': kurtosis, 'skewness': skews, 'max' :maxs, 'min' :mins, 'q5s' :q5s, 'q95s' :q95s, 'q01s':q01s, 'q99s':q99s, 'q10s':q10s, 'q90s': q90s }
#     df = pd.DataFrame(data_dict)
#     # Specify the filename
#     filename = 'pennantModel.xlsx'
#     df.to_excel(filename, index=False)


# else:
#     print("Currently we don't analyze {}".format(app))


