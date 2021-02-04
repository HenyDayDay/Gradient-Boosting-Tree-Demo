"""
A module to reduce the feature space down to a set number of features.

"""

#Setting up the environment
import numpy as np
import pandas as pd
import math
from sklearn import metrics
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def ks_statistic(actual, predictions):
    """
    Calculating the KS statistic:
        https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    
    We will be using the scipy implementation which uses emperical samples
    rather than an assumed distribution
    
    Inputs:
        actual: binary target you are predicting
        predictions: predictions from your model
    Outputs
        KS statistic
    """
    
    #Creating a dataframe for the provided data
    data = pd.DataFrame(data={"target":actual, "prediction":predictions})
    
    #Calculating the score
    ks_score = abs(ks_2samp(data.loc[data.target == 1, "prediction"], data.loc[data.target == 0, "prediction"])[0])
    
    return ks_score

def auc(actual, predictions):
    """
    Calculating the area under the curve (AUC) for the recieving operating chrachteristic curve (ROC):
        https://en.wikipedia.org/wiki/Receiver_operating_characteristic
   
    StatQuest has a nice introduction to how the AUC and ROC curves are derived:
        https://www.youtube.com/watch?v=4jRBRDbJemM
        
    Inputs:
        actual: binary target you are predicting
        predictions: predictions from your model
    Outputs
        AUC metric
    """
    
    #Creating a dataframe for the provided data
    data = pd.DataFrame(data={"target":actual, "prediction":predictions})
    
    #Calculating the score
    auc_score = metrics.roc_auc_score(data["target"], data["prediction"])
    
    return auc_score

def rmse(actual, predictions):
    """
    Calculating the root mean squared error
        https://en.wikipedia.org/wiki/Root-mean-square_deviation
        
    Inputs:
        actual: binary target you are predicting
        predictions: predictions from your model
    Outputs
        RMSE metric
    """
    
    #Creating a dataframe for the provided data
    data = pd.DataFrame(data={"target":actual, "prediction":predictions})
    
    #Calculating the score
    rmse_score = math.sqrt(mean_squared_error(data["target"], data["prediction"]))
    
    return rmse_score

def gini(actual, predictions):
    """
    Calculating the gini coefficient
        https://en.wikipedia.org/wiki/Gini_coefficient
        
    Inputs:
        actual: binary target you are predicting
        predictions: predictions from your model
    Outputs
        gini metric
    """
    
    #Creating a dataframe for the provided data
    gini_coefficient = 2 * auc(actual, predictions) - 1
    
    return gini_coefficient

def quantile_lift(actual, predictions, n_bins=50, lower_percentile=0.05, upper_percentile=0.95):
    """
    Calculating lift of our model lift. To do this we perform the following steps:
        1. Compute the average target by prediction quantiles
        2. Fitting a third-order polynomial to the target vs prediction quantile plot.
        3. Calculate the lift, defined as (95th percentile)/(5th percentile) from our fitted polynomial.
    The reason we use a polynomial is to improve the stability of quantile metric.
        
    Inputs:
        actual: array-like values that correspond to the target when creating the model
        predictions: array-like predictions that correspond to the predictions from the model
    Output:
        lift: defined as (95th percentile)/(5th percentile)
    """
    
    #Creating a dataframe for the provided data
    data = pd.DataFrame(data={"target":actual, "prediction":predictions})

    #Calculating the quantiles by the prediction
    data = data.sort_values(by="prediction", ascending=True)
    data["quantile"] = 1
    data["quantile"] = np.floor(data["quantile"].cumsum()/(data["quantile"].sum()+1e-10)*n_bins)
    data["quantile"] = (data["quantile"] + 0.5)/n_bins 
    
    #Creating the dataframe for the fitting the polynomial
    grouped_data = data.groupby("quantile", as_index=False)["target"].mean()
    
    #Fitting the polynomial
    coef = np.polyfit(grouped_data["quantile"], grouped_data["target"], deg=3)
    
    #Calculating the lift from the fitted polynomial
    lift = np.polyval(coef, upper_percentile)/np.polyval(coef, lower_percentile)
    
    return lift

def mae(actual, predictions):
    """
    Calculating the mean absolute error
        https://en.wikipedia.org/wiki/Mean_absolute_error
        
    Inputs:
        actual: binary target you are predicting
        predictions: predictions from your model
    Outputs
        MAE metric
    """
    
    #Creating a dataframe for the provided data
    data = pd.DataFrame(data={"target":actual, "prediction":predictions})
    
    #Calculating the score
    mae_score = mean_absolute_error(data["target"], data["prediction"])
    
    return mae_score

def calculate_metrics(actual, predictions, decimals=3):
    """
    Calculating a series of metrics for the provided dataset:
        1. KS Statistic
        2. AUC
        3. Quantile Lift
        4. RMSE
        5. Gini
        
    Inputs:
        actual: array-like values that correspond to the target when creating the model
        predictions: array-like predictions that correspond to the predictions from the model
    Output:
        A dictionary containing the metrics list above
    """

    #The dictionary where the metrics will be stored
    metrics_dict = dict()

    #Adding the metrics to the dictionary
    metrics_dict["KS"] = round(ks_statistic(actual, predictions),3)
    metrics_dict["AUC"] = round(auc(actual, predictions),3)
    metrics_dict["Quantile Lift"] = round(quantile_lift(actual, predictions),3)
    metrics_dict["RMSE"] = round(rmse(actual, predictions),3)
    metrics_dict["Gini"] = round(gini(actual, predictions),3)

    return metrics_dict 
    

def k_folds_indices_by_time_period(time_period, index, n_folds, seed):
    """
    A function to create K-Folds indices by holding out long period of time.

    Motivation:
        Financial time-series data includes a lot of feature correlation within time periods that 
        can make the model very unstable in future time periods. It has been found that holding out
        long periods of time rather than random indicies is better for 
    """

    #Storing our data
    data = pd.DataFrame(data={"time_period": time_period, "index": index})
    
    #The unique time periods that we will be sampling from
    unique_time_periods = np.array(data.time_period.unique())

    #The list of dictionaries where we will be storing our indices choices
    k_fold_indicies = []

    #Generating the indices
    for i in range(n_folds):
        indices_dict = dict()

        train_index, test_index = train_test_split(unique_time_periods.copy(), test_size=1/n_folds, random_state=(seed+i))
        train_index, train_valid_index = train_test_split(train_index.copy(), test_size=1/n_folds, random_state=(seed+i))

        indices_dict["train"] = list(data.loc[data.time_period.isin(train_index),"index"])
        indices_dict["valid"] = list(data.loc[data.time_period.isin(train_valid_index),"index"])
        indices_dict["test"] = list(data.loc[data.time_period.isin(test_index),"index"])

        k_fold_indicies.append(indices_dict)

    return k_fold_indicies


def feature_descriptions(data, features):
    """
    A routine to calculate basic information about a feature:
        1. mean - skipping missing values
        2. median - skipping missing values
        3. standard deviation - skipping missing values
        4. 5th percentile - skipping missing values
        5. 95th percentile - skipping missing values
        6. minimum - skipping missing values
        7. maximum - skipping missing values
        8. percent missing values 

    Inputs:
        data: dataframe containing the features
        features: list of columns to calculate the summary statistics
    Output:
        A dataframe of the above metrics, where each row represents 
        feature in the list features
    """

    #Forcing the data to be of type dataframe, since we are assuming it for other types
    assert isinstance(data, pd.DataFrame), "Provided data needs to be of type pandas.DataFrame."

    #Checking the data type of the features
    if isinstance(features, str):
        features = [features]
    assert isinstance(features, list), "Provided features need to either be of type str or list."

    #Copying the dataframe so that we dont alter the original
    data = data.copy()

    #Calculating the metrics
    feature_descriptions = []
    for feature in features:
        mean = round(data[feature].mean(skipna=True),3)
        median = round(data[feature].median(skipna=True),3)
        sigma = round(data[feature].std(skipna=True),3)
        fifth_percentile = round(np.nanpercentile(data[feature],q=5),3)
        ninety_fifth_percentile = round(np.nanpercentile(data[feature], q=95),3)
        minimum = round(data[feature].min(skipna=True),3)
        maximum = round(data[feature].max(skipna=True),3)
        pct_missing = round(sum(data[feature].isnull())/data[feature].shape[0],3)
        feature_descriptions.append([feature, mean, median, sigma, fifth_percentile, ninety_fifth_percentile, minimum, maximum, pct_missing])

    #Clearning up the dataframe    
    feature_descriptions = pd.DataFrame(feature_descriptions)
    feature_descriptions.columns=["feature","mean","median","sigma","5th percentile","95th percentile", "minimum", "maximum", "percent missing"]
    feature_descriptions = feature_descriptions.set_index("feature")

    return feature_descriptions



def summary_statistics(actual, prediction, n_bins=10, n_prediction_intervals=10, sorting="asc"):
    """
    Calculation of statistics grouped by quantile and prediction interval

    The following statistics will be calculated for each
        0. Prediction
        1. Bad Rate
        2. Number of Bads
        3. Number of Goods
        4. Total Loans
        5. Cumulative Bad Pct
        6. Cumulative Good Pct
        7. Cumulative Loan Pct
    """

    def _summarize_df(data):
        """
        Convenience function for calculating the metric for each of the dataframes,
        so that we don't have to repeat the code twice
        """
        data = data.copy()
        data["bad_rate_prediction"] = data["prediction"]/data["observations"]
        data["actual_bad_rate"] = data["actual"]/data["observations"]
        data["bads"] = data["actual"]
        data["goods"] = data["observations"] - data["bads"]
        data["total_loans"] = data["observations"]
        data["total_pct"] = data["observations"]/sum(data["observations"])
        data["bads_pct"] = data["bads"]/sum(data["bads"])
        data["goods_pct"] = data["goods"]/sum(data["goods"])
        data["cumulative_bads_pct"] = data["bads_pct"].cumsum()
        data["cumulative_goods_pct"] = data["goods_pct"].cumsum()
        data["ks_statistic"] = np.absolute(np.round(data["cumulative_goods_pct"] - data["cumulative_bads_pct"], 4) * 100)

        data = data[["bad_rate_prediction", "actual_bad_rate", "bads", "goods","total_loans", "total_pct","bads_pct", "goods_pct", "cumulative_bads_pct", "cumulative_goods_pct", "ks_statistic"]].copy()

        return data


    #Storing the data
    data = pd.DataFrame(data={"actual":actual, "prediction":prediction})
    data["prediction"] = data["prediction"].apply(lambda x: .9999999 if x == 1 else x)
    
    #Creating the quantiles dataset
    quantiles_data = data.sort_values(by="prediction", ascending=True)
    quantiles_data["quantile"] = 1
    quantiles_data["quantile"] = np.floor(quantiles_data["quantile"].cumsum()/(quantiles_data["quantile"].sum()+1e-10)*n_bins) + 1
    quantiles_data["observations"] = 1

    #Grouping the quantiles based data
    quantiles_data = quantiles_data.groupby("quantile")["actual", "prediction", "observations"].sum()

    #Creating the prediction interval dataset
    prediction_interval_data = data.copy()
    prediction_interval_data["prediction_interval"] = np.floor(prediction_interval_data["prediction"]/(1/n_prediction_intervals))
    prediction_interval_data["prediction_interval"] = round(prediction_interval_data["prediction_interval"] / n_prediction_intervals,2).astype(str) + "-" + round((prediction_interval_data["prediction_interval"] + 1) / n_prediction_intervals,2).astype(str)
    prediction_interval_data["observations"] = 1

    #Grouping the prediction interval based data
    prediction_interval_data = prediction_interval_data.groupby("prediction_interval")["actual", "prediction", "observations"].sum()

    #Sorting type
    if sorting == "asc":
        ascending_flag = True
    else:
        ascending_flag = False
    
    #The list where we will be returning our summaries
    summary_statistics = list()

    summary_statistics.append(_summarize_df(quantiles_data).sort_values(by="quantile",ascending=ascending_flag))
    summary_statistics.append(_summarize_df(prediction_interval_data).sort_values(by="prediction_interval",ascending=ascending_flag))

    return summary_statistics 


def str_to_bool(value):
    """
    Converts a string to a boolean value, anything that is now 
    like 'true' or 'false' will be treated as a missing value.
    """
    
    #Typing checking the input
    assert (isinstance(value, str) or (isinstance(value,bool) or (np.isnan(value)))), "The value passed should be of type string or missing value."
    #Nothing to do
    if isinstance(value, bool):
        return value
            
    #Removing upper case value
    if isinstance(value, str):
        value = value.lower()
    
    if value == "true":
        return_value = True
    elif value == "false":
        return_value = False
    else:
        return_value = math.nan

    return return_value

def bool_to_int(value):
    """
    Converts a boolean variable to an int, while retaining missing
    values.
    """
    
    #Typing checking the input
    assert (isinstance(value, bool) or (np.isnan(value))), "The value passed should be of type bool or missing value."
    
    if value == True:
        return_value = 1
    elif value == False:
        return_value = 0
    else:
        return_value = math.nan

    return return_value









    










































