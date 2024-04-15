"""
This script builds a classification model for APPEKG data using shapelet-based time series classification techniques.

Overview:
- The script processes time series data extracted from APPEKG runs, which includes heartbeat
  metrics collected from various threads.
- Each thread's heartbeat metrics are treated as individual time series data point.
- The script first standardizes the length of all time series by padding shorter series with
  zeros to match the longest series.
- It then reduces the dimensionality of these time series using Piecewise Aggregate Approximation (PAA)
  to facilitate more efficient analysis.
  Reference: https://tslearn.readthedocs.io/en/latest/gen_modules/piecewise/tslearn.piecewise.PiecewiseAggregateApproximation.html
- Subsequently, the LearningShapelets method from the tslearn module is employed to identify the
  most representative shapelets, which are crucial features for time series classification.
  Reference: https://tslearn.readthedocs.io/en/latest/gen_modules/shapelets/tslearn.shapelets.LearningShapelets.html
- The script outputs a spreadsheet containing the parameters of the shapelets (number and size),
  F-1 scores, and additional metric data.

Requirements:
- Before running this script, ensure that the optimal shapelet size and number have been
 determined using the `find_optimal_shapelet.py` script.
- The shapelet parameters must be specified before execution.

Execution:
- Run the script from the command line as follows:
  python3 /path/to/train_shapelet.py /path/to/application_runs /path/to/misconfig_runs <app_name> /path/to/optimal_parameters.xlsx"

  optimal_parameters.xlsx: is the spreadsheet taht contains the optimal shapelet parameters for each hb mertic

Directory Structure:
- The 'application_runs' directory for each application should contain subdirectories named 'APPEKG'
  and 'anomalousRun', which hold the respective types of run data.
- The 'misconfig_runs' directory for each application should contain the heartbeat data of misconfigured runs.
  """


from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

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
from tslearn.shapelets import LearningShapelets, \
    grabocka_params_to_shapelet_size_dict
from tensorflow.keras.optimizers.legacy import Adam
from tslearn.piecewise import PiecewiseAggregateApproximation
import tensorflow as tf
from tslearn.utils import to_time_series_dataset
import seaborn as sns
from sklearn.metrics import confusion_matrix



def get_label_from_output(run_directory):
    """
    Determines the label of a run based on keywords found in its output file.
    
    This function searches the specified directory for a file ending with 'out', 
    reads through it, and assigns a label based on specific keywords. The presence 
    of 'good run' or 'normal run' within the file, or 'APPEKG' in the directory name, 
    labels it as 'good'. The presence of 'anomalous run' in the file or 'anomalous' 
    in the directory name labels it as 'anomalous'. Otherwise, it's labeled as 'misconfig'.
    
    Parameters:
    - run_directory (str): The path to the directory containing the output file.
    
    Returns:
    - str: The label determined for the run ('good', 'anomalous', or 'misconfig').
    """
    
    # Default label in case no specific criteria are met
    label = 'misconfig'
    
    # List all files in the given directory
    files = os.listdir(run_directory)
    
    for file_name in files:
        if file_name.endswith('out'):
            file_path = os.path.join(run_directory, file_name)
            
            with open(file_path, "r") as file:
                for line in file:
                    # Check for keywords to determine the label
                    if 'good run' in line or 'normal run' in line or 'APPEKG' in run_directory:
                        label = 'good'
                        break  # Stop reading further if a label is determined
                    elif 'anomalous run' in line or 'anomalous' in run_directory:
                        label = 'anomalous'
                        break  # Stop reading further if a label is determined
            
            # Stop processing other files once a label is determined from one file
            break
            
    return label

def get_number_hbs(dir_paths):
    """
    Counts the number of heartbeat names listed in the first encountered JSON file within the provided directories.

    This function iterates over a list of directory paths, searching for JSON files. Upon encountering the first JSON
    file, it loads the file, retrieves the count of heartbeat names specified under the key "hbnames", and returns this count.
    The search and processing stop as soon as the first JSON file has been processed.

    Parameters:
    - dir_paths (list of str): A list containing directory paths to search for JSON files.

    Returns:
    - int: The number of heartbeat names found in the first encountered JSON file, or 0 if none are found.

    Note:
    - The function returns the count from the first JSON file encountered across all provided directory paths.
    - If no JSON files are found in the given directories, or if "hbnames" key does not exist, it returns 0.
    """
    for dir_path in dir_paths:
        files = os.listdir(dir_path)
        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(dir_path, file_name)
                try:
                    with open(file_path) as fp:
                        data = json.load(fp)
                        # Safely retrieve the count of heartbeat names, defaulting to 0 if not found
                        return len(data.get("hbnames", []))
                except json.JSONDecodeError:
                    # Handle possible JSON decoding error and continue searching
                    print(f"Warning: Could not decode JSON from {file_path}. Skipping.")
                break  # Stop processing further files/directories after the first JSON file
    return 0  # Return 0 if no suitable JSON file was found or processed


def generate_aggregate_data(dir_paths):
    """
    Aggregates and processes heartbeat (HB) count and duration data from CSV files within specified directories.

    Parameters:
    - dir_paths (list): A list of directory paths containing the CSV files to be processed.

    Returns:
    - tuple: Contains lists of HB counts, HB durations, their corresponding labels, means, and the runs they belong to.
    """
    num_hb = get_number_hbs(dir_paths)
    hb_counts, hb_durations, hb_count_means, hb_dur_means = ([] for _ in range(4))
    hb_count_labels, hb_duration_labels, runs = ([] for _ in range(3))

    # Initialize lists to store HB data
    for _ in range(num_hb):
        hb_counts.append([])
        hb_durations.append([])
        hb_count_means.append([])
        hb_dur_means.append([])
        hb_count_labels.append([])
        hb_duration_labels.append([])

    for run in dir_paths:
        files = os.listdir(run)
        label = get_label_from_output(run)
        
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(run, file_name)
                df = pd.read_csv(file_path)
                
                unique_threads = df['threadID'].unique()
                # labels.extend([label] * len(unique_threads))
                
                for hb_index in range(num_hb):
                    for thread_id in unique_threads:
                        runs.append(run)
                        thread_data = df[df['threadID'] == thread_id]

                        # Process HB counts
                        hb_count_data = thread_data[f"hbcount{hb_index + 1}"]
                        hb_count_data_clean = hb_count_data[hb_count_data != 0.0]
                        if not hb_count_data_clean.empty:
                            hb_counts[hb_index].append(hb_count_data.tolist())
                            hb_count_means[hb_index].append(hb_count_data_clean.mean())
                            hb_count_labels[hb_index].append(label)
                        
                        # Process HB durations
                        hb_duration_data = thread_data[f"hbduration{hb_index + 1}"]
                        hb_duration_data_clean = hb_duration_data[hb_duration_data != 0.0]
                        if not hb_duration_data_clean.empty:
                            hb_durations[hb_index].append(hb_duration_data.tolist())
                            hb_dur_means[hb_index].append(hb_duration_data_clean.mean())
                            hb_duration_labels[hb_index].append(label)

    return hb_counts, hb_durations, hb_count_labels, hb_duration_labels, hb_count_means, hb_dur_means, runs 

def list_subdirectories(dir_path):
    """
    Generates a list of subdirectory paths within a given directory, with exceptions.

    This function traverses all subdirectories of the provided directory path. It excludes
    subdirectories that either are clean runs (indicated by "cleanRun" in the path) or belong
    to CoMD runs (indicated by "CoMD" in the path), with a special consideration for CoMD runs
    where the actual directory of interest is one level up from the identified subdirectory.

    Parameters:
    - dir_path (str): The path to the directory whose subdirectories are to be listed.

    Returns:
    - list: A list of paths to the relevant subdirectories, filtered based on specified conditions.
    """
    relevant_paths = []
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # Skip directories without subdirectories
        if not dirs:
            # Skip directories related to clean runs
            if "cleanRun" in root:
                continue
            # Adjust path for CoMD runs to move one directory level up
            if "CoMD" in root:
                root, _ = os.path.split(root)
            relevant_paths.append(root)
    return relevant_paths

def make_data_fit_shapelet(hb_data):
    """
    Prepares heartbeat data for shapelet learning by ensuring uniform length.

    This function converts a list of heartbeat time series into a dataset compatible
    with shapelet learning algorithms. It ensures all time series are of the same length
    by padding shorter ones with zeros. Missing values (NaNs) within the time series are
    also replaced with zeros to maintain data integrity.

    Parameters:
    - hb_data (list of list of float): A list where each element is a heartbeat time series
      represented as a list of floats.

    Returns:
    - np.ndarray: A NumPy array of shape (n_series, series_length) where 'n_series' is the
      number of time series and 'series_length' is the length of the longest series in the
      dataset. Shorter series are padded with zeros.
    """
    # Convert list of time series into a dataset where each time series is of equal length
    hb_data = to_time_series_dataset(hb_data)
    
    # Replace missing values (NaNs) with zeros
    for time_series in hb_data:
        time_series[np.isnan(time_series)] = 0
        
    return hb_data

def pad_series(x, target_length, pad_value=0, pad_start=False):
    """
    Pad a time series to a target length.
    
    Parameters:
    - x: The original time series (1D numpy array).
    - target_length: The desired length of the series.
    - pad_value: The value used for padding. Default is 0.
    - pad_start: If True, padding is added to the start of the series. Otherwise, it's added to the end.
    
    Returns:
    - The padded time series as a 1D numpy array.
    """
    paddedSeries = []
    for series in x:
        padding_length = max(target_length - len(series), 0)
        if pad_start:
            padded_series = np.pad(series, (padding_length, 0), mode='constant', constant_values=pad_value)
            paddedSeries.append(padded_series)
        else:
            padded_series = np.pad(series, (0, padding_length), mode='constant', constant_values=pad_value)
            paddedSeries.append(padded_series)
    
    return paddedSeries

def split_data(data, labels, test_size):
    """
    Splits data into random train and test subsets.

    This function uses scikit-learn's train_test_split method to randomly partition the given data and
    corresponding labels into training and testing sets based on the specified test size. A fixed random
    state is used to ensure reproducible splits each time the function is called.

    Parameters:
    - data (array-like): The input variables/features, structured as an array or matrix.
    - labels (array-like): The target output/labels corresponding to the input data.
    - test_size (float or int): If float, should be between 0.0 and 1.0 and represent the proportion
      of the dataset to include in the test split. If int, represents the absolute number of test samples.

    Returns:
    - X_train (array-like): The subset of input data used for training.
    - X_test (array-like): The subset of input data used for testing.
    - y_train (array-like): The subset of labels used for training.
    - y_test (array-like): The subset of labels used for testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

def reduce_time_series_length(X_train, X_test, segments):
    """
    Reduces the dimensionality of time series data using Piecewise Aggregate Approximation (PAA).

    This function applies PAA to both training and testing time series datasets. It divides each time series
    into a specified number of equally sized segments, computes the mean of each segment, and then assembles
    a new time series from these means. The transformation effectively reduces the time series data's
    dimensionality from its original length to the number of specified segments.

    Parameters:
    - X_train (array-like): The training time series dataset, where each row represents a time series.
    - X_test (array-like): The testing time series dataset, with the same structure as X_train.
    - segments (int): The number of segments to divide each time series into for the PAA transformation.

    Returns:
    - tuple: A tuple containing two elements:
        - The transformed training time series dataset after PAA dimensionality reduction.
        - The transformed testing time series dataset after PAA dimensionality reduction.
    """
    paa = PiecewiseAggregateApproximation(n_segments=segments)
    X_train_transformed = paa.fit_transform(X_train)
    X_test_transformed = paa.fit_transform(X_test)
    
    return X_train_transformed, X_test_transformed


def train_shapelet_classifier(X_train, Y_train, shapelet_sizes, num_iter):
    """
    Trains a shapelet-based classifier on the given time series data.

    This function initializes a LearningShapelets classifier with specified parameters,
    including the number of shapelets per size and the number of iterations for training.
    It uses the Adam optimizer for the learning process. The training duration is measured,
    and both the trained classifier and the total training time are returned.

    Parameters:
    - X_train (array-like): The training data, where each row represents a time series.
    - Y_train (array-like): The target labels for the training data.
    - shapelet_sizes (dict): A dictionary specifying the number of shapelets per size.
    - num_iter (int): The number of iterations to run the training process.

    Returns:
    - tuple: A tuple containing:
        - The trained LearningShapelets classifier.
        - The total time taken to train the classifier, in seconds.
    """
    start_time = time.time()
    
    shapelet_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                     optimizer=tf.optimizers.Adam(0.01),
                                     weight_regularizer=0.01,
                                     batch_size=16,
                                     max_iter=num_iter,
                                     verbose=0,
                                     scale=1,
                                     random_state=42)
    
    shapelet_clf.fit(X_train, Y_train)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return shapelet_clf, total_time

def compute_time_series_means(data):
    """
    Calculates the means of each time series in the dataset, excluding zeros.

    This function iterates through a collection of time series data. For each series,
    it removes any zero values and computes the mean of the remaining values. This approach
    is useful when zeros are not meaningful data points and should not be factored into the
    average computation.

    Parameters:
    - data (list of list of float): A list where each element is a time series represented
      as a list of floats.

    Returns:
    - list: A list of mean values for each time series, with zeros excluded from the calculation.
    """
    means = [np.mean([x for x in series if x != 0]) for series in data]
    return means

def assign_shapelet_distances(n_shapelets, X_train, distances, df, X_misconf, distances_misconf, df_misconf):
    """
    Assigns shapelet distances to the appropriate dataframe columns for both training and misconfiguration data.

    This function iterates over a specified number of shapelets and appends the distances between each shapelet and 
    time series data points to a list. It then adds these lists as new columns to two dataframes: one for training data
    and another for misconfiguration data. Each column in the dataframe represents the distances of the time series data 
    to a specific shapelet.

    Parameters:
    - n_shapelets (int): Number of shapelets to process.
    - X_train (array-like): The training time series dataset.
    - distances (list of list of float): Distances from each time series in X_train to each shapelet.
    - df (pd.DataFrame): The dataframe to which the training distances will be added.
    - X_misconf (array-like): The misconfiguration time series dataset.
    - distances_misconf (list of list of float): Distances from each time series in X_misconf to each shapelet.
    - df_misconf (pd.DataFrame): The dataframe to which the misconfiguration distances will be added.

    Returns:
    None: The function modifies the dataframes in place, adding new columns for each shapelet's distance.
    """
    # Process distances for the training data
    for shapelet_index in range(n_shapelets):
        # Extract distances for the current shapelet across all training samples
        shapelet_distances = [distances[i][shapelet_index] for i, _ in enumerate(X_train)]
        # Add these distances as a new column to the dataframe
        df[f'disShapelet{shapelet_index + 1}'] = shapelet_distances

    # Process distances for the misconfiguration data
    for shapelet_index in range(n_shapelets):
        # Extract distances for the current shapelet across all misconfiguration samples
        shapelet_distances_misconf = [distances_misconf[i][shapelet_index] for i, _ in enumerate(X_misconf)]
        # Add these distances as a new column to the dataframe
        df_misconf[f'disShapelet{shapelet_index + 1}'] = shapelet_distances_misconf
    

def plot_shapelet_time_series_distances(df, n_shapelets, hb_name, app_name, df_predicted):
    """
    Plots the distances from time series to shapelets, comparing actual labels to predicted ones.

    This function generates scatter plots for each shapelet, showing the distances from time series to that shapelet.
    Points are colored based on their actual label ('good' or 'anomalous') and the predicted classification
    ('good' or 'anomalous' as misclassified). The plots are saved as PNG files.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the actual labels and distances to shapelets.
    - n_shapelets (int): The number of shapelets used in the analysis.
    - hb_name (str): The name of the heartbeat variable being analyzed.
    - app_name (str): The name of the application the analysis is focused on.
    - df_predicted (pd.DataFrame): Dataframe containing the predicted classifications and distances to shapelets.

    Returns:
    None: The function saves the plots as PNG files and does not return any value.
    """
    labels = df["labels"]
    predictions = df_predicted[hb_name + ' predicted']
    actual_means = df[hb_name + 'Mean']
    predicted_means = df_predicted[hb_name + 'Mean']

    for k in range(n_shapelets):
        shapelet_distance = 'disShapelet' + str(k+1)
        x_actual = df[shapelet_distance]
        x_predicted = df_predicted[shapelet_distance]

        # Separating actual and predicted good and anomalous based on labels
        x_good, y_good = x_actual[labels == 'good'], actual_means[labels == 'good']
        x_bad, y_bad = x_actual[labels != 'good'], actual_means[labels != 'good']
        x_pred_good, y_pred_good = x_predicted[predictions == 'good'], predicted_means[predictions == 'good']
        x_pred_bad, y_pred_bad = x_predicted[predictions != 'good'], predicted_means[predictions != 'good']

        plt.figure(figsize=(10, 6))
        plt.scatter(x_good, y_good, color='g', label='Good')
        plt.scatter(x_bad, y_bad, color='r', label='Anomalous')
        plt.scatter(x_pred_good, y_pred_good, color='#C6E14D', label='Misclassified Good')
        plt.scatter(x_pred_bad, y_pred_bad, color='#DE8410', label='Misclassified Anomalous')
        plt.xlabel(f'Distance (time series t, shapelet-{k+1})')
        plt.ylabel(f'{hb_name} Mean')
        plt.legend(loc='best')
        plt.title(f'{app_name} {hb_name} - Distance to Shapelet {k+1}')
        plt.savefig(f'{app_name}_{hb_name}_ShapeletDistanceWithMean{k+1}.png')
        plt.close()




def main():
    if len(sys.argv) != 5:
        print("Usage: python train_shapelet.py /path/to/application_runs /path/to/misconfig_runs <app_name> optimal_parameters.xlsx")
        sys.exit(1)

    app_runs_path = sys.argv[1]
    misconfig_runs_path = sys.argv[2]
    app_name = sys.argv[3]
    # Specify the path to your Excel file
    file_path = sys.argv[4]
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Aggregate data from application runs and misconfiguration runs
    app_data_paths = list_subdirectories(app_runs_path)
    misconfig_data_paths = list_subdirectories(misconfig_runs_path)

    hb_counts, hb_durations, hbc_labels,hbd_abels, hbc_means, hbd_means, runs = generate_aggregate_data(app_data_paths)
    misconfig_hb_counts, misconfig_hb_durations, misconfig_hbc_labels, misconfig_hbd_abels, misconfig_hbc_means, misconfig_hbd_means,\
         misconfig_runs = generate_aggregate_data(misconfig_data_paths)

    number_of_hbs = get_number_hbs(app_data_paths)
    columns = ['Metric', 'Shapelet Size', 'F1-Score', 'Accuracy','Training Time', \
        'original time series length',  'PAAed time series length', '# training samples', '# test samples']
    
    df_summury = pd.DataFrame(columns=columns)

    for i in range(number_of_hbs):

        misconfig_df = pd.DataFrame()
        hb_count_df =pd.DataFrame()
        misc_hb_count_df =pd.DataFrame()
        misc_hb_duration_df =pd.DataFrame()
        hb_duration_df =pd.DataFrame()
        print(f'processing hbcount{i+1}')
        hb_count_data = hb_counts[i]
        misconfig_hb_count_data = misconfig_hb_counts[i]
        labels = hbc_labels[i]
        hb_count_data = make_data_fit_shapelet(hb_count_data)
        # make misconfig data same data model length
        target_length = hb_count_data.shape[1]
        padded_misc_hb_count_data = pad_series(misconfig_hb_count_data, target_length, pad_value=0, pad_start=False)
        padded_misc_hb_count_data_arr = np.array(padded_misc_hb_count_data)
        paa = PiecewiseAggregateApproximation(n_segments=1000)
        reduced_misconf_hb_count = paa.fit_transform(padded_misc_hb_count_data_arr)
        X_train1, X_test1, y_train1, y_test1 = split_data(hb_count_data, labels, 0.5) 
        hb_count_means = compute_time_series_means(X_test1)
        misc_hb_count_means = compute_time_series_means(misconfig_hb_count_data)
        hb_name = 'HB' + str(i+1) + '-Count'
        misc_hb_count_df[hb_name + "Mean"] = misc_hb_count_means
        original_tseries_len = X_train1.shape[1]
        X_train1, X_test1 = reduce_time_series_length(X_train1, X_test1, 1000)
        PAAed_tseries_len = X_train1.shape[1]
        hb_count_df[hb_name + "Mean"] = hb_count_means
        X_train1 = TimeSeriesScalerMinMax().fit_transform(X_train1)
        X_test1 = TimeSeriesScalerMinMax().fit_transform(X_test1)
        hb_count_df['labels'] = y_test1
        num_shapelet = df.loc[df['Heartbeat'] == f'hbCount{i+1}', 'Optimal_Num_Shapelets'].values[0]
        shaplete_size = df.loc[df['Heartbeat'] == f'hbCount{i+1}', 'Optimal_Shapelet_Size'].values[0]
        shapelet_sizes = {int(shaplete_size): int(num_shapelet)}
        hbCount_shp_clf, time1 = train_shapelet_classifier(X_train1, y_train1, shapelet_sizes, 800)
        predicted = hbCount_shp_clf.predict(reduced_misconf_hb_count)
        misconfig_df[hb_name + 'predicted class'] = predicted
        misc_hb_count_df[hb_name + ' predicted'] = predicted
        distances = hbCount_shp_clf.transform(X_test1)
        hb_count_misc_dis = hbCount_shp_clf.transform(reduced_misconf_hb_count)
        nshapelets = len(hbCount_shp_clf.shapelets_)
        plt.figure()
        for s, shapelet in enumerate(hbCount_shp_clf.shapelets_):
            shapelet = TimeSeriesScalerMinMax().fit_transform(shapelet.reshape(1, -1, 1)).flatten()
            plt.subplot(len(hbCount_shp_clf.shapelets_), 1, s + 1)
            plt.plot(shapelet.ravel())
            plt.title(f'Shapelet {s}')
            plt.savefig(app_name + hb_name + 'Shapelets' + str(s) + 'Plot.png')
            plt.close()
            with open(app_name + hb_name + 'Shapelets' + str(s) + '.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['shapelet' + str(s)])
                writer.writerow([shapelet])
        ypred = hbCount_shp_clf.predict(X_test1)
        hb_count_df['predicted'] = ypred
        f1 = f1_score(y_test1, ypred, average='macro', pos_label='anomalous')
        new_row = {'Metric': hb_name, 'Shapelet Size':shapelet_sizes, \
        'F1-Score': f1, 'Accuracy': hbCount_shp_clf.score(X_test1, y_test1), 'Training Time': time1, \
        'original time series length': original_tseries_len, 'PAAed time series length': PAAed_tseries_len,\
            '# training samples': X_train1.shape[0], '# test samples': X_test1.shape[0]}
        df_summury = df_summury.append(new_row, ignore_index=True)
        assign_shapelet_distances(nshapelets, X_test1, distances, hb_count_df, reduced_misconf_hb_count, hb_count_misc_dis, misc_hb_count_df)
        plot_shapelet_time_series_distances(hb_count_df, nshapelets, hb_name, app_name, misc_hb_count_df)
        output_file = app_name + hb_name + 'shapelets.xlsx'
        hb_count_df.to_excel(output_file)
        misc_hb_count_df.to_excel(app_name + hb_name + 'predicted_misconfig.xlsx')
        print(f'finshed hbcount{i+1}')

        print(f'processing hbduration{i+1}')
        hb_duration_data = hb_durations[i]
        misconfig_hb_duration_data = misconfig_hb_durations[i]
        labels = hbd_abels[i]
        hb_duration_data = make_data_fit_shapelet(hb_duration_data)
        # make misconfig data same data model length
        target_length = hb_duration_data.shape[1]
        padded_misc_hb_duration_data = pad_series(misconfig_hb_duration_data, target_length, pad_value=0, pad_start=False)
        padded_misc_hb_duration_data_arr = np.array(padded_misc_hb_duration_data)
        paa = PiecewiseAggregateApproximation(n_segments=1000)
        reduced_misconf_hb_duration = paa.fit_transform(padded_misc_hb_duration_data_arr)
        X_train1, X_test1, y_train1, y_test1 = split_data(hb_count_data, labels, 0.5) 
        hb_duration_means = compute_time_series_means(X_test1)
        misc_hb_duration_means = compute_time_series_means(misconfig_hb_duration_data)
        hb_name = 'HB' + str(i+1) + '-Duration'
        misc_hb_duration_df[hb_name + "Mean"] = misc_hb_duration_means
        original_tseries_len = X_train1.shape[1]
        X_train1, X_test1 = reduce_time_series_length(X_train1, X_test1, 1000)
        PAAed_tseries_len = X_train1.shape[1]
        hb_duration_df[hb_name + "Mean"] = hb_duration_means
        X_train1 = TimeSeriesScalerMinMax().fit_transform(X_train1)
        X_test1 = TimeSeriesScalerMinMax().fit_transform(X_test1)
        hb_duration_df['labels'] = y_test1
        num_shapelet = df.loc[df['Heartbeat'] == f'hbDuration{i+1}', 'Optimal_Num_Shapelets']
        shaplete_size = df.loc[df['Heartbeat'] == f'hbDuration{i+1}', 'Optimal_Shapelet_Size']
        shapelet_sizes = {int(shaplete_size): int(num_shapelet)}
        hbduration_shp_clf, time1 = train_shapelet_classifier(X_train1, y_train1, shapelet_sizes, 800)
        predicted = hbduration_shp_clf.predict(reduced_misconf_hb_duration)
        misconfig_df[hb_name + 'predicted class'] = predicted
        misc_hb_duration_df[hb_name + ' predicted'] = predicted
        distances = hbduration_shp_clf.transform(X_test1)
        hb_duration_misc_dis = hbduration_shp_clf.transform(reduced_misconf_hb_duration)
        nshapelets = len(hbduration_shp_clf.shapelets_)
        plt.figure()
        for s, shapelet in enumerate(hbduration_shp_clf.shapelets_):
            shapelet = TimeSeriesScalerMinMax().fit_transform(shapelet.reshape(1, -1, 1)).flatten()
            plt.subplot(len(hbduration_shp_clf.shapelets_), 1, s + 1)
            plt.plot(shapelet.ravel())
            plt.title(f'Shapelet {s}')
            plt.savefig(app_name + hb_name + 'Shapelets' + str(s) + 'Plot.png')
            plt.close()
            with open(app_name + hb_name + 'Shapelets' + str(s) + '.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['shapelet' + str(s)])
                writer.writerow([shapelet])
        ypred = hbduration_shp_clf.predict(X_test1)
        hb_duration_df['predicted'] = ypred
        f1 = f1_score(y_test1, ypred, average='macro', pos_label='anomalous')
        new_row = {'Metric': hb_name, 'Shapelet Size':shapelet_sizes, \
        'F1-Score': f1, 'Accuracy': hbduration_shp_clf.score(X_test1, y_test1), 'Training Time': time1, \
        'original time series length': original_tseries_len, 'PAAed time series length': PAAed_tseries_len,\
            '# training samples': X_train1.shape[0], '# test samples': X_test1.shape[0]}
        df_summury = df_summury.append(new_row, ignore_index=True)
        assign_shapelet_distances(nshapelets, X_test1, distances, hb_duration_df, reduced_misconf_hb_duration, hb_duration_misc_dis, misc_hb_duration_df)
        plot_shapelet_time_series_distances(hb_duration_df, nshapelets, hb_name, app_name, misc_hb_duration_df)
        output_file = app_name + hb_name + 'shapelets.xlsx'
        hb_duration_df.to_excel(output_file)
        misc_hb_duration_df.to_excel(app_name + hb_name + 'predicted_misconfig.xlsx')
        print(f'finshed hbduration{i+1}')
    df_summury.to_excel(app_name + '_shapelets.xlsx')

    
if __name__ == "__main__":
    main()





