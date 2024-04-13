
"""
This Python script is a comprehensive data analysis tool that performs linear regression, 
correlation analysis, and classification of data points based on their labels. It is designed 
to work with datasets containing quality metrics and classifications, specifically focusing on 
applications related to "heartbeats" identified by their means in column names. Key 
functionalities include counting occurrences of specific keywords in DataFrame column names, 
fitting a linear regression model, calculating Pearson correlation coefficients, categorizing 
input data into classes based on labels, and plotting linear regression results with categorized 
data points. Additionally, it constructs a DataFrame summarizing metrics for heartbeat data, 
which includes intercepts, coefficients, and correlation values for both counts and durations of 
heartbeat metrics.

Run the Script: Navigate to the directory containing the script and execute it with Python, 
providing the path to your dataset and an application name as arguments. For example:

python3 linearReg.py /path/to/your_dataset.xlsx YourApplicationName

"""



import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import linear_model
import re


 
 


def get_number_of_hbs(df, keyword=r"\bhb\d+c mean\b"):
    """
    Counts the occurrences of a specified keyword in the DataFrame column names.

    This function iterates through the column names of the provided DataFrame, counting
    how many times a specified keyword appears.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns are to be searched.
    - keyword (str): The keyword to search for in the column names. Default is "hb".

    Returns:
    - int: The number of columns containing the keyword, representing the number of heartbeats.

    Example:
    >>> df = pd.DataFrame(columns=["hb1c_mean", "hb1_std", "hb2_mean", "hb2_std"])
    >>> print(get_number_of_hbs(df))
    2
    """
    num_hb = 0
    for col in df.columns:
        match = re.search(keyword, col)
        if match:
            num_hb+=1
    return num_hb


# Define the function linearRegressionModel with x (features) and y (target) as input parameters
def fit_linear_regression(x, y):
    """
    Fits a linear regression model to the given feature and target data, prints and returns the intercept 
    and first coefficient of the model, along with the predicted target values.

    Parameters:
    - x : array-like or 2D array
        Independent variable(s) or feature data. Shape should be (n_samples, n_features) 
        where n_samples is the number of samples and n_features is the number of features.
    - y : array-like
        Dependent variable or target data. Shape should be (n_samples,) for single output regressions.

    Returns:
    - intercept : float
        The intercept of the regression line (the expected mean value of Y when all X=0).
    - coef : float
        The first coefficient of the regression model. Represents the change in the target 
        associated with a one-unit change in the first feature, holding other features constant.
    - y_pred : array
        Predicted target values for the input features X, based on the fitted linear regression model.

   """
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    intercept = reg.intercept_[0]
    coef = reg.coef_[0][0] 
    y_pred = reg.predict(x)
    
    return(intercept, coef, y_pred)

def find_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two arrays.
    
    This function computes the Pearson correlation coefficient between two numpy arrays. The Pearson correlation coefficient measures the linear relationship between two datasets. It is a value between -1 and 1 where 1 means a perfect positive linear relationship, -1 means a perfect negative linear relationship, and 0 means no linear relationship between the datasets.
    
    Parameters:
    - x (numpy.ndarray): A 1D numpy array representing the first dataset.
    - y (numpy.ndarray): A 1D numpy array representing the second dataset.
    
    Returns:
    float: The Pearson correlation coefficient between x and y.
    
    Example:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([5, 4, 3, 2, 1])
    >>> find_correlation(x, y)
    -1.0
    
    Note:
    Both x and y must have the same length.
    
    Raises:
    ValueError: If x and y have different lengths.
    """ 
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")    
    corr = np.corrcoef(x, y)
    return corr[0][1]


def find_classes_lists(x, y):
    """
    Categorizes input data into three classes ("good", "misconfig", and "bad") based on their labels.

    This function separates data points into different categories based on a classification provided by the `labels` list. Each element in the `labels` list corresponds to the classification of the data points in `x` and `y` at the same index. The function categorizes data into "good", "misconfig", or "bad" based on these labels, and returns separate lists for `x` and `y` values associated with each category.

    Parameters:
    - x : list
        A list of data points. This could represent a feature in a dataset.
    - y : list
        A list of data points corresponding to `x`. This could represent another feature or the target variable associated with `x`.
    - labels : list
        A list of strings that classify each pair of data points in `x` and `y`. Each string should be "good", "misconfig", or any other value considered as "bad".

    Returns:
    - x_good, x_bad, x_mis : lists
        Lists of data points from `x` categorized as "good", "bad", and "misconfig", respectively.
    - y_good, y_bad, y_mis : lists
        Lists of data points from `y` categorized as "good", "bad", and "misconfig", respectively.

    Example:
    >>> x = [1, 2, 3, 4]
    >>> y = [10, 20, 30, 40]
    >>> labels = ["good", "bad", "misconfig", "good"]
    >>> find_classes_lists(x, y, labels)
    ([1, 4], [2], [10, 40], [20], [3], [30])

    Note:
    The function assumes that `x`, `y`, and `labels` are all the same length and properly aligned, meaning each index corresponds across the three lists.
    """
    x_good = []
    y_good = []
    x_bad = []
    y_bad = []
    x_mis = []
    y_mis = []
    for i, label in enumerate(labels):
        if label == "good":
            x_good.append(x[i])
            y_good.append(y[i])
        elif label == "misconfig":
            x_mis.append(x[i])
            y_mis.append(y[i])
        else:
            x_bad.append(x[i])
            y_bad.append(y[i])
    return x_good, x_bad, y_good, y_bad, x_mis, y_mis

def plot_linear_reg(x_good, x_bad, y_good, y_bad, y_pred, hb_name, x_mis, y_mis):
    """
    Plot linear regression results along with data points categorized into Good Run, Bad Run, and Misconfigured.

    This function visualizes the relationship between a quality metric and a specified variable (hb_name) through linear regression, displaying Good Runs, Bad Runs, and Misconfigurations as distinct scatter points.

    Parameters:
    - x_good (array-like): The x-coordinates of the Good Run data points.
    - x_bad (array-like): The x-coordinates of the Bad Run data points.
    - y_good (array-like): The y-coordinates of the Good Run data points corresponding to x_good.
    - y_bad (array-like): The y-coordinates of the Bad Run data points corresponding to x_bad.
    - y_pred (array-like): The y-values predicted by the linear regression model across the x-axis.
    - hb_name (str): The name of the variable being analyzed, used as the y-axis label and in the filename for saving the plot.
    - x_mis (array-like): The x-coordinates of the Misconfigured data points.
    - y_mis (array-like): The y-coordinates of the Misconfigured data points corresponding to x_mis.
    
    Returns:
    None: The function does not return any values but saves and displays the plot.
    
    Example:
    >>> plot_linear_reg([1,2,3], [4,5,6], [7,8,9], [10,11,12], [5,6,7,8,9,10], "HBDu", [1.5, 2.5, 3.5], [7.5, 8.5, 9.5])
    # This will create and display a scatter plot with linear regression line.
    
    Note:
    - The plot is saved as "<applicationName><variableName>LinearReg.png" in the current working directory.
    - The function requires `matplotlib.pyplot` for plotting.
    """
    plt.plot(x, y_pred, color="k")  # Assuming x should be a combined list of all x-values
    plt.xlabel("Quality")
    plt.ylabel(hb_name)
    plt.scatter(x_good, y_good, color="g", label="GoodRun")
    plt.scatter(x_bad, y_bad, color="r", label="BadRun")
    plt.scatter(x_mis, y_mis, color="orange", label="MisConfig")
    plt.legend()
    plt.savefig(app_name + hb_name + "LinearReg.png")
    plt.show()
    plt.close()
    
def create_metric_table(interceptsCounts, interceptsDurs, coefsCounts, coefsDurs, correlationsCounts, correlationsDurs, hb_names, numHBs):
    """
    Constructs a DataFrame summarizing metrics for heartbeat (HB) data.

    This function aggregates intercepts, coefficients, and correlation values for counts and durations of heartbeat data, organizing them into a structured table for analysis.

    Parameters:
    - interceptsCounts (list of float): Intercept values for counts across different HBs.
    - interceptsDurs (list of float): Intercept values for durations across different HBs.
    - coefsCounts (list of float): Coefficient values for counts across different HBs.
    - coefsDurs (list of float): Coefficient values for durations across different HBs.
    - correlationsCounts (list of float): Correlation values for counts across different HBs.
    - correlationsDurs (list of float): Correlation values for durations across different HBs.
    - hb_names (list of str): Names of the heartbeat metrics.
    - numHBs (int): Number of heartbeats being considered.

    Returns:
    pd.DataFrame: A DataFrame where each row corresponds to a heartbeat metric, including its name, correlation values, coefficients, and intercepts for both counts and durations.

    Example:
    >>> interceptsCounts = [1.1, 2.2]
    >>> interceptsDurs = [3.3, 4.4]
    >>> coefsCounts = [0.1, 0.2]
    >>> coefsDurs = [0.3, 0.4]
    >>> correlationsCounts = [0.9, 0.8]
    >>> correlationsDurs = [0.7, 0.6]
    >>> hb_names = ["HB1", "HB2"]
    >>> numHBs = 2
    >>> df = create_metric_table(interceptsCounts, interceptsDurs, coefsCounts, coefsDurs, correlationsCounts, correlationsDurs, hb_names, numHBs)
    >>> print(df)
    
    Notes:
    - The function assumes lists for intercepts, coefficients, and correlations are pre-calculated and provided as input.
    - The function is designed for paired count and duration metrics for each HB, requiring input list lengths to be consistent with "numHBs".
    """
    df = pd.DataFrame()
    slopes = []
    corrs = []
    coeffs = []

    # Aggregate values for each heartbeat into lists
    for hb in range(numHBs):
        slopes.extend([interceptsCounts[hb], interceptsDurs[hb]])
        coeffs.extend([coefsCounts[hb], coefsDurs[hb]])
        corrs.extend([correlationsCounts[hb], correlationsDurs[hb]])

    # Assign aggregated lists to DataFrame columns
    df["Metric"] = hb_names  # Assumes hb_names needs to be repeated for counts and durations
    df["Correlation"] = corrs
    df["Coeff"] = coeffs
    df["Intercept"] = slopes
    return df

def plot_regression_and_categories(x, y_pred, categories, hb_name):
    """
    Plots linear regression results alongside categorized data points.

    This function creates a scatter plot for each category of data ('good', 'bad', 'misconfig') with different colors
    and plots the linear regression line that fits the data. It visualizes the relationship between two variables
    and highlights how data points are categorized. The plot is saved as an image file named after the application
    and the specific heartbeat variable being analyzed.

    Parameters:
    - x (array-like): The x-coordinates for the data points, typically representing the independent variable.
    - y_pred (array-like): The predicted y-values from the linear regression model, used to plot the regression line.
    - categories (dict): A dictionary containing tuples of x and y values for each category ('good', 'bad', 'misconfig').
      Each key is a string indicating the category, and each value is a tuple of two lists (x_values, y_values).
    - hb_name (str): The name of the heartbeat variable being analyzed, used in the plot title and the filename for saving.

    Returns:
    None: This function does not return a value. It generates and displays a plot, and saves the plot as a PNG file.
    """
    plt.figure(figsize=(10, 6))
    colors = {'good': 'green', 'bad': 'red', 'misconfig': 'orange'}
    for category, (x_vals, y_vals) in categories.items():
        plt.scatter(x_vals, y_vals, color=colors[category], label=category.capitalize() + ' Run')
    plt.plot(x, y_pred, color='black', label='Regression Line')
    plt.xlabel('Quality')
    plt.ylabel(hb_name)
    plt.title(f'Linear Regression for {hb_name}')
    plt.legend()
    plt.savefig(f'{app_name}_{hb_name}_LinearRegression.png')
    plt.show()
    
def categorize_data(x, y, labels):
    """
    Categorizes data into 'good', 'misconfig', and 'bad' based on provided labels.

    This function iterates over provided data points and their corresponding labels, grouping the data points
    into three categories: 'good', 'misconfig', and 'bad'. Each category is associated with lists of x and y values
    that belong to that category. This categorization facilitates further analysis and visualization.

    Parameters:
    - x (list): A list of x-values representing the independent variable in the dataset.
    - y (list): A list of y-values representing the dependent variable in the dataset.
    - labels (list): A list of strings representing the category labels for each data point. Expected labels are
      'good', 'misconfig', or any other label, which will be categorized as 'bad'.

    Returns:
    tuple of tuples: A tuple containing three tuples, each representing a category ('good', 'bad', 'misconfig').
      Each inner tuple contains two lists: the first list contains the x-values, and the second list contains
      the y-values for the data points in that category.
    """
    categories = {'good': ([], []), 'misconfig': ([], []), 'bad': ([], [])}
    for value, label in zip(zip(x, y), labels):
        categories[label if label in categories else 'bad'][0].append(value[0])
        categories[label if label in categories else 'bad'][1].append(value[1])
    return categories['good'], categories['bad'], categories['misconfig']

def generate_metric_summary(intercepts, coefficients, correlations, hb_names):
    """
    Generates a summary DataFrame containing metrics for heartbeat data analysis.

    This function compiles the intercepts, coefficients, and correlation values obtained from linear regression
    analysis into a pandas DataFrame. Each row in the DataFrame corresponds to a different heartbeat variable,
    with columns for the variable name ('Metric'), intercept, coefficient, and correlation value.

    Parameters:
    - intercepts (list): A list of intercept values from the linear regression models.
    - coefficients (list): A list of coefficient values from the linear regression models.
    - correlations (list): A list of Pearson correlation coefficients between the independent and dependent variables.
    - hb_names (list): A list of strings representing the names of the heartbeat variables analyzed.

    Returns:
    pd.DataFrame: A DataFrame where each row corresponds to a heartbeat variable and contains its intercept,
      coefficient, and correlation value. The DataFrame has columns labeled 'Metric', 'Intercept', 'Coefficient',
      and 'Correlation'.
    """
    return pd.DataFrame({
        'Metric': hb_names,
        'Intercept': intercepts,
        'Coefficient': coefficients,
        'Correlation': correlations
    })

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py /path/to/dataset.xlsx ApplicationName")
        sys.exit(1)

    file_path, app_name = sys.argv[1:3]
    df = pd.read_excel(file_path)

    # Analyze and visualize the data
    intercepts, coefficients, correlations, hb_names = [], [], [], []
    for col in df.filter(regex=r"\bhb\d+[cd] mean\b").columns:
        x = df[['Qualities']].values
        y = df[[col]].values
        intercept, coefficient, y_pred = fit_linear_regression(x, y)
        correlation = find_correlation(df['Qualities'], df[col])
        
        intercepts.append(intercept)
        coefficients.append(coefficient)
        correlations.append(correlation)
        hb_names.append(col)
        
        good, bad, misconfig = categorize_data(df['Qualities'], df[col], df['Classes'])
        plot_regression_and_categories(df['Qualities'], y_pred, {'good': good, 'bad': bad, 'misconfig': misconfig}, col)

    summary_df = generate_metric_summary(intercepts, coefficients, correlations, hb_names)
    summary_df.to_excel(f'{app_name}_MetricsSummary.xlsx')

# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python3 script.py /path/to/file.xlsx application_name")
#         sys.exit(1)

#     file_path, app_name = sys.argv[1:3]
#     df = pd.read_excel(file_path)
#     num_hbs = get_number_of_hbs(df)

#     hb_count_intercpt = []
#     hb_count_coef = []
#     hb_dur_intercpt = []
#     hb_dur_coef = []
#     hb_count_corr = []
#     hb_dur_corr = []
#     hb_names = []
#     for hb in range(num_hbs):
#         hb_name = "hb" + str(hb+1) + "c"
#         hb_names.append(hb_name)
#         x = np.array(df["Qualities"]).reshape(-1,1)
#         y = np.array(df[hb_name + " mean"]).reshape(-1,1)
#         labels = np.array(df["Classes"])
#         intercept, coef, y_pred = linearRegressionModel(x, y)
#         hb_count_coef.append(coef)
#         hb_count_intercpt.append(intercept)
#         x = np.array(df["Qualities"])
#         y = np.array(df[hb_name + " mean"])
#         corrltion = find_correlation(x, y)
#         hb_count_corr.append(corrltion)
#         x_good, x_bad, y_good, y_bad, x_mis, y_mis = find_classes_lists(x, y)
#         hb_name = "HB" + str(hb+1) + "-Count"
#         plot_linear_reg(x_good, x_bad, y_good, y_bad, y_pred, hb_name, x_mis, y_mis)
    

#         hb_name = "hb" + str(hb+1) + "d"
#         hb_names.append(hb_name)
#         x = np.array(df["Qualities"]).reshape(-1,1)
#         y = np.array(df[hb_name + " mean"]).reshape(-1,1)
#         labels = np.array(df["Classes"])
#         intercept, coef, y_pred = linearRegressionModel(x, y)
#         hb_dur_intercpt.append(intercept)
#         hb_dur_coef.append(coef)
#         x = np.array(df["Qualities"])
#         y = np.array(df[hb_name  + " mean"])
#         corrltion = find_correlation(x, y)
#         hb_dur_corr.append(corrltion)
#         x_good, x_bad, y_good, y_bad, x_mis, y_mis = find_classes_lists(x, y)
#         hb_name = "HB" + str(hb+1) + "-Duration"
#         plot_linear_reg(x_good, x_bad, y_good, y_bad, y_pred, hb_name, x_mis, y_mis)


#     df3 = create_metric_table(hb_count_intercpt, hb_dur_intercpt, hb_count_coef,  hb_dur_coef, hb_count_corr , hb_dur_corr, hb_names, num_hbs)
#     df3.to_excel(app_name + "linearRegCorr.xlsx")