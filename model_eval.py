"""
A module to evaluate model through plotting and metrics.

Can perform the following:
    1. Plot actual vs predicted with equal observations in each group
    2. Plot actual vs predicted with equal response variable bin length
    3. Double lift charts to compare two models.
    4. AUC chart
"""

#Setting up the environment
import pandas as pd
import numpy as np
from scipy.special import gammaln
from numba import jit
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import Range1d, LinearAxis
from bokeh.palettes import Spectral11, BuPu3, Spectral4
from bokeh.layouts import row
from sklearn import metrics
from scipy.stats import ks_2samp

#Loading Bokeh
output_notebook()

def _plot_dark_mode(p):
    """
    "Draws Everything using dark mode
    """
    p.title.text_font_size = "16pt"
    p.title.text_color = "#e8e8e8"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    
    p.background_fill_color = "#292d3a"
    p.border_fill_color = "#292d3a"
        
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.xaxis.axis_label_text_color ="#e8e8e8"
    p.yaxis.axis_label_text_color ="#e8e8e8"
    p.xaxis.axis_line_color = "#e8e8e8"
    p.yaxis.axis_line_color = "#e8e8e8"
    p.xaxis.major_label_text_color = "#e8e8e8"
    p.yaxis.major_label_text_color = "#e8e8e8"
    p.xaxis.major_tick_line_color = "#e8e8e8"
    p.xaxis.minor_tick_line_color = "#e8e8e8"
    p.yaxis.major_tick_line_color = "#e8e8e8"
    p.yaxis.minor_tick_line_color = "#e8e8e8"
        
    p.legend.background_fill_color = "#292d3a"
    p.legend.background_fill_alpha = .2
    p.legend.border_line_color = "#e8e8e8"
    p.toolbar_location="above"
    p.legend.label_text_color = "#e8e8e8"
    p.legend.label_text_font_style = "italic"
        
    p.xgrid.grid_line_color = "#e8e8e8"
    p.xgrid.grid_line_dash = [3,4] 
    p.ygrid.grid_line_color = "#e8e8e8"
    p.ygrid.grid_line_dash = [3,4]
    p.legend.background_fill_color = "#292d3a"
    p.legend.border_line_color = "#e8e8e8"
    p.legend.label_text_color = "#e8e8e8"
    p.legend.label_text_font_style = "italic"

def _plot_light_mode(p):
    """
    "Draws Everything using dark mode
    """
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
        
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
        
    p.legend.background_fill_alpha = .2
    p.toolbar_location="above"
    p.legend.label_text_font_style = "italic"
        
    p.xgrid.grid_line_dash = [3,4] 
    p.ygrid.grid_line_dash = [3,4]
    p.legend.label_text_font_style = "italic"

    
def actual_vs_predicted(actual, predicted, actual_name="Target", predicted_name="Prediction", y_axis_label="Target", 
                        plot_1_x_axis_label="Quantile (Equal Observations Per Quantile)", plot_2_x_axis_label="Quantile (Equal Length Prediction Interval)", plot_label="Actual vs Predicted", observations_name="Observations", n_bins=50, dark_mode=True, normalize=True, filter_low_observations=True):
    """
    We will plot the actual vs predicted with equal observations in each group, also
    with the target variable split into equal length groupings.
    
    Inputs:
        actual: target variable used in training the model
        predicted: the predicted variable from the model
        actual_name: label for plotting
        predicted_name: label for plotting
        n_bins: number of quantiles
        
    """
    
    #Creating a dataframe for our dataset
    data = pd.DataFrame(data={"target":np.array(actual), "prediction":np.array(predicted)})
    
    #To account for trend issues, we can normalize the data
    if normalize:
        data["prediction"] = data["prediction"]*(sum(data["target"])/sum(data["prediction"]))
        
        
    """
    Plot 1 - Actual vs Predicted plot with equal observations in each quantile.
    """
    
    #Calculating the quantiles
    data = data.sort_values(by="prediction")
    data["quantile"] = 1
    data["quantile"] = np.floor(data["quantile"].cumsum()/(data["quantile"].sum()+1e-10)*n_bins)
    
    #Calculating the actual vs predicted by quantile
    grouped_data = data.groupby("quantile", as_index=False)["target","prediction"].mean()
    grouped_data["observations"] = data.groupby("quantile", as_index=False)["prediction"].count()["prediction"]
    grouped_data["quantile"] = grouped_data.index
    
    #Calculating the ranges for the first plot
    exposure_max = float(np.max(grouped_data.observations)) * 1.2
    y_max = np.max([grouped_data.target, grouped_data.prediction]) * 1.2
    y_min = np.min([grouped_data.target, grouped_data.prediction]) / 1.2
    
    #Figure creation
    p1 = figure(width=600, height=400, title=plot_label, 
               x_axis_label=plot_1_x_axis_label, y_axis_label=y_axis_label, 
               y_range=[y_min,y_max])
    
    #Adding the right hand axes
    p1.extra_y_ranges = {"Observations": Range1d(start=0, end=exposure_max)}
    p1.add_layout(LinearAxis(y_range_name="Observations", axis_label = observations_name), 'right')
    
    #Setting the line colors depending upon whether or not it is light mode or dark mode
    if dark_mode:
        line_colors = [Spectral11[1],Spectral11[3]]
        observations_color = Spectral11[3]
    else:
        line_colors = [Spectral4[0],Spectral4[3]]
        observations_color = Spectral4[0]


    #Drawing Everything
    p1.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.prediction.values),
           line_width=3, line_color=line_colors[0], legend=predicted_name)
    
    p1.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.target.values),
           line_width=3, line_color=line_colors[1], legend=actual_name)
    
    x_range = grouped_data["quantile"].values
    left_range = list(np.array(x_range)-0.9*(x_range[1] - x_range[0])/2)
    right_range =  list(np.array(x_range)+0.9*(x_range[1] - x_range[0])/2)
    p1.quad(top=list(grouped_data["observations"].values), 
           left=left_range,
           right=right_range, 
           bottom=0, alpha=0.2, 
           color = observations_color, 
           y_range_name="Observations")
    
    p1.right[0].formatter.use_scientific = False
    
    p1.min_border_right = 100
    
    #Adding dark mode
    if dark_mode:
        _plot_dark_mode(p1)
    else:
        _plot_light_mode(p1)
        
    """
    Plot 2 - Actual vs Predicted plot with equal length bins for the response.
    """    
        
    #Calculating the quantiles
    max_target = np.max(data["target"])
    min_target = np.min(data["target"])
    bin_length = (max_target - min_target)/n_bins
    data["quantile"] = (np.floor(data["prediction"]/bin_length) + 0.5) * bin_length
    
    #Calculating the actual vs predicted by quantile
    grouped_data = data.groupby("quantile", as_index=False)["target","prediction"].mean()
    grouped_data["observations"] = data.groupby("quantile", as_index=False)["prediction"].count()["prediction"]
    if filter_low_observations:
        grouped_data = grouped_data[grouped_data["observations"] > 25]
    
    #Calculating the ranges for the first plot
    exposure_max = float(np.max(grouped_data.observations)) * 1.2
    y_max = np.max([grouped_data.target, grouped_data.prediction]) * 1.2
    y_min = np.min([grouped_data.target, grouped_data.prediction]) / 1.2
    
    #Figure creation
    p2 = figure(width=600, height=400, title=plot_label, 
               x_axis_label=plot_2_x_axis_label, y_axis_label=y_axis_label, 
               y_range=[y_min,y_max])
    
    #Adding the right hand axes
    p2.extra_y_ranges = {"Observations": Range1d(start=0, end=exposure_max)}
    p2.add_layout(LinearAxis(y_range_name="Observations", axis_label = observations_name), 'right')
    
    #Drawing Everything
    p2.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.prediction.values),
           line_width=3, line_color=line_colors[0], legend=predicted_name)
    
    p2.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.target.values),
           line_width=3, line_color=line_colors[1], legend=actual_name)
    
    x_range = grouped_data["quantile"].values
    left_range = list(np.array(x_range)-0.9*(x_range[1] - x_range[0])/2)
    right_range =  list(np.array(x_range)+0.9*(x_range[1] - x_range[0])/2)
    p2.quad(top=list(grouped_data["observations"].values), 
           left=left_range,
           right=right_range, 
           bottom=0, alpha=0.2, 
           color = observations_color, 
           y_range_name="Observations")
    p2.right[0].formatter.use_scientific = False
    
    p2.min_border_left=100
    
    if dark_mode:
        _plot_dark_mode(p2)
    else:
        _plot_light_mode(p2)

    show(row(p1,p2))
    
def double_lift(model_1_pred, model_2_pred, actual, model_1_name="Model_1", model_2_name="Model_2", actual_name="Actual",
                x_axis_label="Percent Difference in Predictions (Model_2/Model_1)", y_axis_label="Target", plot_label="Double Lift", 
                n_bins=50, dark_mode=True, normalize=True):
    """
    Plotting the double lift chart to compare the two models. Explanation of the double lift:
        1. A ratio of the predictions is computed, let r = model_2_pred/model_1_pred
        2. r is grouped and ranked from low to high with equal observation in each group
        3. r is plotted along the x_axis and the model_1_pred, model_2_pred, actual are plotted along 
           y_axis
    This type of comparison allows us to investigate which model is correct when they differ.
    
    Inputs:
        model_1_pred: The predictions from the first model
        model_2_pred: The predictions from the second model
        actual: The target variable from the models
        model_1_name: The name of the first model in the plot
        model_2_name: The name of the second model in the plot
        actual_name: The name of the target variable in the plot
        x_axis_label: The name of the ratio between the two models
        y_axis_label: The name of the target you are predicting
        n_bins: Number of quantiles used for grouping the ratio of the two models
        dark_mode=dark mode plotting if true, light mode plotting if false
    
    Output:
        double lift plot
        metrics on the two models
    """
    
    #Creating a dataframe for our dataset
    data = pd.DataFrame(data={"target":np.array(actual), "model_1":np.array(model_1_pred), "model_2":np.array(model_2_pred)})
    
    #To account for trend issues, we can normalize the data
    if normalize:
        data["model_1"] = data["model_1"]*(sum(data["target"])/sum(data["model_1"]))
        data["model_2"] = data["model_2"]*(sum(data["target"])/sum(data["model_2"]))
    
    #The ratio of the two model predictions that we will be using
    data["ratio"] = data["model_2"]/data["model_1"]
    
    #Creating the bins
    data = data.sort_values(by="ratio", ascending=True)
    data["quantiles"] = 1
    data["quantile"] = np.floor(data["quantiles"].cumsum()/(data["quantiles"].sum()+1e-10)*n_bins)
    
    #Calculating that values by group
    grouped_data = data.groupby("quantile", as_index=False)["target","model_1", "model_2","ratio"].mean()
    grouped_data["observations"] = data.groupby("quantile", as_index=False)["model_1"].count()["model_1"]
    grouped_data = grouped_data[grouped_data.ratio.isin([np.inf, -np.inf]) == False]
    
    #Calculating the ranges for the plot
    exposure_max = float(np.max(grouped_data.observations)) * 1.2
    y_max = np.max([grouped_data.target, grouped_data.model_1, grouped_data.model_2]) * 1.2
    y_min = np.min([grouped_data.target, grouped_data.model_1, grouped_data.model_2]) / 1.2
    
    #Figure creation
    p1 = figure(width=600, height=400, title=plot_label, 
               x_axis_label=x_axis_label, y_axis_label=y_axis_label, 
               y_range=[y_min,y_max])
    
    #Adding the right hand axes
    p1.extra_y_ranges = {"Observations": Range1d(start=0, end=exposure_max)}
    p1.add_layout(LinearAxis(y_range_name="Observations", axis_label = "Observations"), 'right') 

    #Setting the line colors depending upon whether or not it is light mode or dark mode
    if dark_mode:
        line_colors = [Spectral11[1],Spectral11[2], Spectral11[4]]
        observations_color = Spectral11[4]
    else:
        line_colors = [Spectral4[0],Spectral4[3],Spectral4[1]]
        observations_color = Spectral4[0]
    
    #Drawing Everything
    p1.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.model_1.values),
           line_width=3, line_color=line_colors[0], legend=model_1_name)
    
    p1.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.model_2.values),
           line_width=3, line_color=line_colors[1], legend=model_2_name)
    
    p1.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.target.values),
           line_width=3, line_color=line_colors[2], legend=actual_name)
    
    x_range = grouped_data["quantile"].values
    left_range = list(np.array(x_range)-0.9*(x_range[1] - x_range[0])/2)
    right_range =  list(np.array(x_range)+0.9*(x_range[1] - x_range[0])/2)
    p1.quad(top=list(grouped_data["observations"].values), 
           left=left_range,
           right=right_range, 
           bottom=0, alpha=0.2, 
           color = observations_color, 
           y_range_name="Observations")
    p1.right[0].formatter.use_scientific = False    
    
    if dark_mode:
        _plot_dark_mode(p1)
    else:
        _plot_light_mode(p1)
        
    show(p1)
    
def one_way_percentile_plot(feature, actual, predicted, actual_name="Target", predicted_name="Prediction", y_axis_label="Target", 
                plot_1_x_axis_label="Percentile",  plot_label="Actual vs Predicted", 
                observations_name="Observations", n_bins=50, dark_mode=True, normalize=False):
    """
    We will plot the actual vs predicted for the percentiles of a feature, also
    with the target variable split into equal length groupings.
    
    Inputs:
        actual: target variable used in training the model
        predicted: the predicted variable from the model
        actual_name: label for plotting
        predicted_name: label for plotting
        n_bins: number of quantiles
        
    """
    
    #Creating a dataframe for our dataset
    data = pd.DataFrame(data={"feature":feature,"target":np.array(actual), "prediction":np.array(predicted)})
    data = data[data.feature.isnull() == False]

    #Make sure there is enough values for the number of percentiles
    if len(data.feature.unique()) < n_bins:
        n_bins = len(data.feature.unique())
    
    #To account for trend issues, we can normalize the data
    if normalize:
        data["prediction"] = data["prediction"]*(sum(data["target"])/sum(data["prediction"]))
        
        
    """
    Plot 1 - Actual vs Predicted plot with equal observations in each quantile.
    """
    
    #Calculating the quantiles
    data = data.sort_values(by="feature")
    data["quantile"] = 1
    data["quantile"] = np.round(np.floor(data["quantile"].cumsum()/(data["quantile"].sum()+1e-10)*n_bins)/n_bins + 0.5/n_bins,3)
    
    #Calculating the actual vs predicted by quantile
    grouped_data = data.groupby("quantile", as_index=False)["target","prediction"].mean()
    grouped_data["observations"] = data.groupby("quantile", as_index=False)["prediction"].count()["prediction"]
    
    #Calculating the ranges for the first plot
    exposure_max = float(np.max(grouped_data.observations)) * 1.2
    y_max = np.max([grouped_data.target, grouped_data.prediction]) * 1.2
    y_min = np.min([grouped_data.target, grouped_data.prediction]) / 1.2
    
    #Figure creation
    p1 = figure(width=600, height=400, title=plot_label, 
               x_axis_label=plot_1_x_axis_label, y_axis_label=y_axis_label, 
               y_range=[y_min,y_max])
    
    #Adding the right hand axes
    p1.extra_y_ranges = {"Observations": Range1d(start=0, end=exposure_max)}
    p1.add_layout(LinearAxis(y_range_name="Observations", axis_label = observations_name), 'right')
    
    #Setting the line colors depending upon whether or not it is light mode or dark mode
    if dark_mode:
        line_colors = [Spectral11[1],Spectral11[3]]
        observations_color = Spectral11[3]
    else:
        line_colors = [Spectral4[0],Spectral4[3]]
        observations_color = Spectral4[0]


    #Drawing Everything
    p1.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.prediction.values),
           line_width=3, line_color=line_colors[0], legend=predicted_name)
    
    p1.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.target.values),
           line_width=3, line_color=line_colors[1], legend=actual_name)
    
    x_range = grouped_data["quantile"].values
    left_range = list(np.array(x_range)-0.9*(x_range[1] - x_range[0])/2)
    right_range =  list(np.array(x_range)+0.9*(x_range[1] - x_range[0])/2)
    p1.quad(top=list(grouped_data["observations"].values), 
           left=left_range,
           right=right_range, 
           bottom=0, alpha=0.2, 
           color = observations_color, 
           y_range_name="Observations")
    
    p1.right[0].formatter.use_scientific = False
    
    p1.min_border_right = 100
    
    #Adding dark mode
    if dark_mode:
        _plot_dark_mode(p1)
    else:
        _plot_light_mode(p1)
    
    show(p1)

def sensitivity(feature, actual, predicted, response_name, y_axis_label="Target", 
                plot_1_x_axis_label="Percentile", plot_label="Sensitivity", 
                n_bins=50, dark_mode=True, normalize=False, toolbar=False):
    """
    We will plot the actual vs predicted for the percentiles of a feature, also
    with the target variable split into equal length groupings.
    
    Inputs:
        actual: target variable used in training the model
        predicted: the predicted variable from the model
        actual_name: label for plotting
        predicted_name: label for plotting
        n_bins: number of quantiles
        
    """
    
    #Creating a dataframe for our dataset
    data = pd.DataFrame(data={"feature":feature,"target":np.array(actual), "prediction":np.array(predicted)})
    data = data[data.feature.isnull() == False]
    
    #To account for trend issues, we can normalize the data
    if normalize:
        data["prediction"] = data["prediction"]*(sum(data["target"])/sum(data["prediction"]))
        
        
    """
    Plot 1 - Actual vs Predicted plot with equal observations in each quantile.
    """
    
    #Average change
    avg_response = data.target.mean()
    
    #Calculating the quantiles
    data = data.sort_values(by="feature")
    if len(data.feature.unique()) < n_bins:
        data["quantile"] = data["feature"]
    else:
        data["quantile"] = 1
        data["quantile"] = np.floor(data["quantile"].cumsum()/(data["quantile"].sum()+1e-10)*n_bins).astype(int)
        
        bin_length = 100/n_bins
        percentiles_to_calculate = np.arange(bin_length/2,100,bin_length)
        feature_percentiles = np.sort(np.nanpercentile(data["feature"], percentiles_to_calculate, interpolation='linear'))
        percentiles_df = pd.DataFrame(data={"quantile_2":feature_percentiles})
        percentiles_df["quantile"] = np.array(percentiles_df.index).astype(int)
        
        data = data.merge(percentiles_df, on="quantile", how="left")
        data["quantile"] = data["quantile_2"]
    
    #Calculating the actual vs predicted by quantile
    grouped_data = data.groupby("quantile", as_index=False)["target","prediction", "feature"].sum()
    grouped_data["observations"] = data.groupby("quantile", as_index=False)["prediction"].count()["prediction"]
    grouped_data = grouped_data.groupby("quantile", as_index=False)["target","prediction", "feature", "observations"].sum()
    grouped_data["target"] = grouped_data["target"]/grouped_data["observations"]
    grouped_data["prediction"] = grouped_data["prediction"]/grouped_data["observations"]
    grouped_data["quantile"] = grouped_data["quantile"].astype(int)
    grouped_data = grouped_data.sort_values(by="quantile")

    
    #Calculating the ranges for the first plot
    exposure_max = float(np.max(grouped_data.observations)) * 1.2
    y_max = np.max([grouped_data.target, grouped_data.prediction]) * 1.2
    y_min = np.min([grouped_data.target, grouped_data.prediction]) / 1.2

    #Figure creation
    p1 = figure(width=600, height=400, title=plot_label, 
               x_axis_label=plot_1_x_axis_label, y_axis_label=y_axis_label, 
               y_range=[y_min,y_max])
    
    #Setting the line colors depending upon whether or not it is light mode or dark mode
    if dark_mode:
        line_colors = [Spectral11[1],Spectral11[3]]
        observations_color = Spectral11[3]
    else:
        line_colors = [Spectral4[0],Spectral4[3]]
        observations_color = Spectral4[0]
    
    p1.line(x=list(grouped_data["quantile"].values), 
           y=list(grouped_data.target.values),
           line_width=3, line_color=line_colors[0], legend=response_name)
    p1.line(x=[grouped_data["quantile"].min(), grouped_data["quantile"].max()],
            y=[avg_response,avg_response],
            line_width=3, line_color=line_colors[1], legend = ("Avg " + response_name),
            line_dash = "dashed") 
    p1.circle(x=list(grouped_data["quantile"].values), y=list(grouped_data.target.values),size=9, fill_color="white")
    
    if dark_mode:
        _plot_dark_mode(p1)
    else:
        _plot_light_mode(p1)
    
    if toolbar == False:
        p1.toolbar.logo = None
        p1.toolbar_location = None
    
    return p1
    
    
def pdp_plot(model, data, feature_label, title, x_axis_plot_label, y_axis_plot_label, n_percentiles=50, toolbar=False):
    """
    A model technique agnostic function to calculate the partial dependence plot of a
    feature.
    
    Inputs:
        model: model object with a predict() method
        data: data for making predictions
        feature_label: the feature we want to calculate the pdp plot for
        x_axis_plot_label: self-explanatory 
        y_axis_plot_label: self-explanatory
    Outputs:
        Plot of the pdp with 2 sigma variance
        
    Notes:
        Missing values for the feature are ignored.
    """
    
    data = data.copy()
    
    #We will be showing every 5 percent
    if len(data[feature_label].unique()) < n_percentiles:
        feature_percentiles = np.sort(np.array(data[feature_label].unique()))
    else:
        bin_length = 100/n_percentiles
        percentiles_to_calculate = np.arange(bin_length/2,100,bin_length)
        feature_percentiles = np.nanpercentile(data[feature_label], percentiles_to_calculate, interpolation='linear')
    
    #Storing the mean response and the variance
    feature_mean_response = list()
    feature_variance = list()
    
    #looping through the percentiles to get the mean and variance in the prediction
    for percentile in feature_percentiles:
        data[feature_label] = percentile
        translated_data = model.predict(data)
        feature_mean_response.append(np.mean(translated_data))
        feature_variance.append(np.var(translated_data))
    
    #Creating the upoper and lower limits
    feature_upper_limit = list(np.array(feature_mean_response) + np.array(feature_variance))
    feature_lower_limit = list(np.array(feature_mean_response) - np.array(feature_variance))
    
    #Plotting everything
    p1 = figure(width=600, height=400, title = title, x_axis_label=x_axis_plot_label, y_axis_label=y_axis_plot_label)
    line_colors = [Spectral4[0],Spectral4[3]]
    p1.line(x=feature_percentiles,y=feature_mean_response,line_width=3, line_color=line_colors[0], legend="Mean Prediction")
    p1.line(x=feature_percentiles,y=feature_upper_limit,line_width=3, line_alpha=0.4, line_dash="dashed", line_color=line_colors[0], legend="Plus/Minus Sigma")
    p1.line(x=feature_percentiles,y=feature_lower_limit,line_width=3,  line_alpha=0.4, line_dash="dashed", line_color=line_colors[0])
    p1.circle(x=feature_percentiles, y=feature_mean_response,size=9, fill_color="white")
    
    _plot_light_mode(p1)
    if toolbar == False:
        p1.toolbar.logo = None
        p1.toolbar_location = None
    
    return(p1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
