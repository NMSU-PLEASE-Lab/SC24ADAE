"""
Script Description:

This script is designed to optimize shapelet parameters—specifically, the number and size of shapelets—used for building a time series classification model based on APPEKG data.

Overview:

Data Processing: The script begins by extracting time series data from APPEKG runs, where each thread’s heartbeat metrics are considered as individual data points.
Standardization: It standardizes the time series lengths by padding shorter series with zeros to equalize them to the length of the longest series in the dataset.
Dimensionality Reduction: The script applies Piecewise Aggregate Approximation (PAA) to reduce the dimensionality of the time series, enhancing the efficiency of the subsequent analysis.
PAA Documentation
Shapelet Optimization: It employs the LearningShapelets method from the tslearn module, using cross_val_score to evaluate different shapelet configurations. The script selects the smallest parameters that yield the maximum cross-validation score.
LearningShapelets Documentation
Output: Results are saved in a spreadsheet detailing the optimal shapelet parameters (number and size) for each analyzed metric.

Execution Instructions:

Run the script from the command line by navigating to the script's directory and typing the following command:
    python3 train_shapelet.py /path/to/application_runs <app_name>
Directory Structure:

Ensure the 'application_runs' directory contains appropriately named subdirectories ('APPEKG' and 'anomalousRun') for each application, which house the respective types of run data.


"""

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


def get_run_label(run_directory):
    """
    Extracts and returns the label of a computational run by inspecting its output files.
    
    This function searches through a specified directory for output files, reading their content
    to determine whether the run is classified as 'good' or 'anomalous'. The classification is
    extracted from predefined keywords within the files. Initial classification can also be
    derived from specific substrings in the directory name itself.

    Parameters:
    - run_directory : str
        The path to the directory that contains the output files of the run.

    Returns:
    - label : str
        The determined label of the run ('good', 'anomalous', or 'undetermined' if no
        conclusive indicators are found within the files).

    Raises:
    - FileNotFoundError: If the specified directory does not exist.
    - IOError: If files within the directory cannot be opened or read.

    Example:
    - Calling get_run_label('/path/to/run_output') might return 'good' if the output
      files or directory name indicate a successful run.
    """
    
    # Attempt to classify based on directory name hints (PENNANT)
    label = 'undetermined'
    if 'APPEKG' in run_directory:
        label = 'good'
    elif 'anomalousRuns' in run_directory:
        label = 'anomalous'
    
    # Attempt to list files in the given directory
    try:
        files = os.listdir(run_directory)
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory {run_directory} does not exist.")

    # Read through each file to find run classification
    for fileName in files:
        if fileName.endswith('.out'):  # Target specific output files
            filePath = os.path.join(run_directory, fileName)
            
            # Open and read the file safely
            try:
                with open(filePath, "r") as file:
                    for line in file:
                        if 'good run' in line:
                            return 'good'
                        elif 'anomalous run' in line:
                            return 'anomalous'
            except IOError:
                raise IOError(f"Error reading file {filePath}.")
            
            # Break after the first output file is processed to prevent overwriting of label
            break

    return label

def get_number_hbs(directory_path):
    """
    Counts and returns the number of heartbeat names (HBs) listed in the first JSON file encountered
    in the provided directory path. This function is designed to parse files generated by APPEKG, 
    which outputs heartbeat names under the key 'hbnames' in a JSON format.

    Parameters:
    - directory_path : list
        A list of directory paths where the JSON files are expected to be found.

    Returns:
    - numberHB : int
        The number of heartbeat names found in the first JSON file encountered. If no JSON file is found,
        the function returns 0.

    Raises:
    - FileNotFoundError: If any listed directory does not exist or no JSON files are found.
    - json.JSONDecodeError: If the JSON file is not properly formatted.

    Example:
    - numberHB = get_number_hbs(['/path/to/directory'])
      This might return 5 if the first JSON file in '/path/to/directory' contains five heartbeat names.
    """
    
    numberHB = 0  # Default return value if no HB names are found
    json_found = False

    for run_directory in directory_path:
        # Check if the directory exists
        if not os.path.exists(run_directory):
            raise FileNotFoundError(f"The directory {run_directory} does not exist.")

        # List files in the directory
        files = os.listdir(run_directory)
        
        # Process each file in the directory
        for fileName in files:
            if fileName.endswith(".json"):
                file_path = os.path.join(run_directory, fileName)
                json_found = True

                # Safely open and read the JSON file
                with open(file_path, 'r') as file:
                    try:
                        data = json.load(file)
                        numberHB = len(data["hbnames"])
                        break  # Exit after processing the first JSON file
                    except json.JSONDecodeError:
                        raise json.JSONDecodeError(f"Error decoding JSON in file {file_path}")

        if json_found:
            break  # Exit after the first directory containing a JSON file

    if not json_found:
        raise FileNotFoundError("No JSON files were found in the provided directories.")

    return numberHB


def generate_aggregate_data(directory_path):
    """
    Processes CSV files within each subdirectory of a given directory path to generate aggregate data for heartbeat counts and durations. Each CSV file is assumed to represent a run containing multiple threads.

    The function aggregates heartbeat counts and durations per thread and calculates mean values across these for each heartbeat identifier across all runs.

    Parameters:
    - directory_path : list
        List of paths to directories containing CSV files for processing.

    Returns:
    - tuple : (hb_counts, hb_durations, hb_count_labels, hb_duration_labels, hb_count_means, hb_dur_means, runs)
        Where each element in the tuple is a list of lists containing data aggregated across runs for each heartbeat identifier.

    Notes:
    - Each CSV file should contain columns named 'threadID', 'hbcount{i}', and 'hbduration{i}' where {i} is a heartbeat identifier.
    - The function assumes that all heartbeat count and duration data are non-negative.
    """
    num_hb = get_number_hbs(directory_path)  # Expected to be a predefined function returning number of heartbeats to process
    hb_counts, hb_durations = [], []
    hb_count_labels, hb_duration_labels = [], []
    hb_count_means, hb_dur_means = [], []
    runs = []

    # Initialize lists to collect data for each heartbeat
    for i in range(num_hb):
        hb_counts.append([])
        hb_durations.append([])
        hb_count_labels.append([])
        hb_duration_labels.append([])
        hb_count_means.append([])
        hb_dur_means.append([])
    labels = []

    # Process each directory path provided
    for run in directory_path:
        files = os.listdir(run)
        label = get_run_label(run)  # Assumes a function that fetches label of run
        
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(run, file_name)
                df = pd.read_csv(file_path)
                
                for d in df['threadID'].unique():
                    labels.append(label)  # Collect label for each unique thread ID

                for hb in range(num_hb):
                    hb_count_col = f"hbcount{hb+1}"
                    hb_duration_col = f"hbduration{hb+1}"

                    if hb_count_col in df.columns and hb_duration_col in df.columns:
                        for d in df['threadID'].unique():
                            runs.append(run)                      
                            hb_count_data = df.loc[df['threadID'] == d, hb_count_col].dropna().tolist()
                            hb_duration_data = df.loc[df['threadID'] == d, hb_duration_col].dropna().tolist()
                            
                            if hb_count_data:
                                mean_count = sum(hb_count_data) / len(hb_count_data)
                                hb_count_means[hb].append(mean_count)
                                hb_counts[hb].append(hb_count_data)
                                hb_count_labels[hb].append(label)

                            if hb_duration_data:
                                mean_duration = sum(hb_duration_data) / len(hb_duration_data)
                                hb_dur_means[hb].append(mean_duration)
                                hb_durations[hb].append(hb_duration_data)
                                hb_duration_labels[hb].append(label)

    return (hb_counts, hb_durations, hb_count_labels, hb_duration_labels, hb_count_means, hb_dur_means, runs)

def list_sub_directories(dir_path):
    """
    Lists directories corresponding to 'APPEKG' and 'anomalousRuns' runs within a given directory path,
    excluding directories related to 'clean runs', 'newAnalysis', and 'ICs'.

    This function searches through all subdirectories of the specified path, filtering out certain
    directories based on specific naming conventions and adding relevant paths to the results.

    Parameters:
    - dir_path : str
        The path to the directory from which subdirectories will be listed.

    Returns:
    - list of str
        A list containing the paths of directories that are either 'APPEKG' or 'anomalousRuns' runs.
    
    Notes:
    - This function skips directories that are deemed 'clean' (having "cleanRuns" in their path),
      along with "newAnalysis" and "ICs".
    - It also adjusts paths if they include directories related to 'CoMD' runs.
    """
    relevant_paths = []
    excluded_keywords = ["cleanRuns", "newAnalysis", "ICs"]
    comd_related = ["galaxy", "parameterfiles", "ICs", "CoMD"]

    for root, dirs, files in os.walk(dir_path, topdown=False):
        # Filter out excluded directories
        if any(keyword in root for keyword in excluded_keywords):
            continue
        
        # Adjust path for CoMD related directories
        if any(keyword in root for keyword in comd_related):
            root, _ = os.path.split(root)
        
        relevant_paths.append(root)

    return relevant_paths

def make_data_fit_shapelet(hb_data):
    """
    Pads all time series in the dataset to the length of the longest time series. This normalization
    is necessary to prepare data for shapelet learning algorithms, which require input time series
    of uniform length.

    Parameters:
    - hb_data : array-like
        A list or array of time series data where each time series may have a different length.

    Returns:
    - numpy.ndarray
        A numpy array of shape (n_series, max_len), where n_series is the number of time series and
        max_len is the length of the longest time series in the dataset. Shorter time series are
        padded with zeros.

    Notes:
    - This function converts the input data into a tslearn-compatible time series dataset.
    - Missing values (NaNs) within any time series are replaced with zeros.
    """
    # Convert the list of time series to a tslearn-compatible time series dataset
    hb_data = to_time_series_dataset(hb_data)

    # Pad shorter time series with zeros to ensure uniform length
    for t in hb_data:
        t[np.isnan(t)]=0

    return hb_data

def split_data(data, labels, test_size):
    """
    Splits the provided data into training and testing subsets.

    This function wraps the sklearn `train_test_split` method to partition data and its corresponding labels
    into train and test subsets, based on a specified proportion for the test set.

    Parameters:
    - data : array-like, shape (n_samples, n_features)
        The input data to be split. Each row corresponds to a sample, and each column to a feature.
    - labels : array-like, shape (n_samples,)
        The target labels associated with the data.
    - test_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.

    Returns:
    - X_train : array-like, shape (n_train_samples, n_features)
        The subset of the data used for training.
    - X_test : array-like, shape (n_test_samples, n_features)
        The subset of the data used for testing.
    - y_train : array-like, shape (n_train_samples,)
        The subset of the labels used for training.
    - y_test : array-like, shape (n_test_samples,)
        The subset of the labels used for testing.

    Example:
    >>> data = [[0, 1], [2, 3]]
    >>> labels = [0, 1]
    >>> split_data(data, labels, test_size=0.5)
    (array([[0, 1]]), array([[2, 3]]), array([0]), array([1]))
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

def reduce_time_series_length(X_train, X_test, n_segments):
    """
    Reduces the dimensionality of time series data in the training and testing sets using 
    Piecewise Aggregate Approximation (PAA). This transformation involves dividing each 
    time series into `n_segments` equal parts (if possible), computing the mean of each 
    segment, and using these means to represent the original time series.

    Parameters:
    - X_train : array-like, shape (n_samples, n_timestamps)
        Training data consisting of multiple time series.
    - X_test : array-like, shape (n_samples, n_timestamps)
        Testing data consisting of multiple time series.
    - n_segments : int
        The number of segments to divide each time series into, which reduces the time 
        series length to this number of points.

    Returns:
    - X_train_paa : array-like, shape (n_samples, n_segments)
        Transformed training data with reduced dimensionality.
    - X_test_paa : array-like, shape (n_samples, n_segments)
        Transformed testing data with reduced dimensionality.

    Example:
    >>> from tslearn.generators import random_walks
    >>> X_train = random_walks(n_ts=50, sz=256, d=1)
    >>> X_test = random_walks(n_ts=20, sz=256, d=1)
    >>> X_train_paa, X_test_paa = reduce_time_series_length(X_train, X_test, 10)
    """
    # Initialize the PiecewiseAggregateApproximation transformer with the specified number of segments
    paa = PiecewiseAggregateApproximation(window_size=n_segments)
    
    # Transform the training data
    X_train_paa = paa.fit_transform(X_train)
    
    # Transform the testing data
    X_test_paa = paa.fit_transform(X_test)
    
    return X_train_paa, X_test_paa


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



def main():
    if len(sys.argv) < 3:
        print(f"Error: missing the runs path\nPlease use: python {sys.argv[0]} </path/to/app/runs> <application name>")
        return
    
    app_path = sys.argv[1]
    app_name = sys.argv[2]
    directories = list_sub_directories(app_path)
    hb_counts, hb_durations, hb_count_labels, hb_duration_labels, hb_count_means, hb_duration_means, runs = generate_aggregate_data(directories)

    # Get number of heartbeats
    num_hbs = get_number_hbs(directories)

    for i in range(num_hbs):
        # Process hbcount data
        data = make_data_fit_shapelet(hb_counts[i])
        data = data.reshape(data.shape[0], data.shape[1])
        X_train, X_test, y_train, y_test = split_data(data, hb_count_labels[i], 0.7)
        X_train, X_test = reduce_time_series_length(X_train, X_test, 2)
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)
        
        # Finding optimal shapelets for hbcount
        optimal_num_shapelets, optimal_shapelet_size, scores = find_optimal_shapelets(X_train, y_train, 
                                                                                      num_shapelets_range=[1, 2, 3], 
                                                                                      shapelet_size_range=[30, 50, 100, 200], 
                                                                                      cv=2)
        print(f"hbCount{i+1}: optimal_num_shapelets {optimal_num_shapelets}, optimal_shapelet_size {optimal_shapelet_size}, scores: {scores}")
        results = results.append({
            'Heartbeat': f'hbCount{i+1}',
            'Optimal_Num_Shapelets': optimal_num_shapelets,
            'Optimal_Shapelet_Size': optimal_shapelet_size
        }, ignore_index=True)

        # Process hbduration data similarly...
        data = make_data_fit_shapelet(hb_counts[i])
        data = data.reshape(data.shape[0], data.shape[1])
        X_train, X_test, y_train, y_test = split_data(data, hb_count_labels[i], 0.7)
        X_train, X_test = reduce_time_series_length(X_train, X_test, 2)
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)
        
        # Finding optimal shapelets for hbcount
        optimal_num_shapelets, optimal_shapelet_size, scores = find_optimal_shapelets(X_train, y_train, 
                                                                                      num_shapelets_range=[1, 2, 3], 
                                                                                      shapelet_size_range=[30, 50, 100, 200], 
                                                                                      cv=2)
        print(f"hbDuration{i+1}: optimal_num_shapelets {optimal_num_shapelets}, optimal_shapelet_size {optimal_shapelet_size}, scores: {scores}")
        results = results.append({
            'Heartbeat': f'hbDuration{i+1}',
            'Optimal_Num_Shapelets': optimal_num_shapelets,
            'Optimal_Shapelet_Size': optimal_shapelet_size
        }, ignore_index=True)
    results.to_excel(app_name + '_optimal_shapelet.xlsx', index=False)

if __name__ == "__main__":
    main()

