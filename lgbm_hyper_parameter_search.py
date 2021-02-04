"""
A module to construct LightGBM models. For more details, visit: 
https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf

This module will perform the following:
    1. Hyperparameter searching using cross validation techniques
    2. Training final models
    3. Creation of PDP plots in either 1 or 2 dimensions.
    4. Importance Scores in 1 or 2 dimensions.
    5. Prediction on new datasets.
    6. Explain of variable contribution for a single prediction.

Our module will search for hyperparameters using Bayesian hyperparameter optimization:
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf

"""

#Setting up the environment
import numpy as np
import pandas as pd
import sys
import random 
import lightgbm as lgb
import hyperopt
from hyperopt import STATUS_OK, hp, Trials
from functools import partial
from scipy.stats import ks_2samp
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
import copy
import time
from basic_functions import *

def _fit_lgbm_model(params, X_train, y_train, X_valid=None, y_valid=None):
    """
    Creating a standard function for fitting a LGBM model.

    Inputs:
        Params: A dictionary of the parameters for fitting the LGBM model.
                Should be noted that the number of trees value is floated and
                will be determined by the validation set.

       Output:
           A lightgbm model
    """    
    
    #Creating a copy of the parameters
    fit_params = params.copy()
    
    #Creating the datasets 
    train_data = lgb.Dataset(data=X_train, label=y_train)

    #Type conversions
    fit_params["max_depth"] = int(fit_params["max_depth"])
    fit_params["num_leaves"] = int(2**fit_params["max_depth"])
    fit_params["min_data_in_leaf"] = int(fit_params["min_data_in_leaf"])

    #Putting the parameters in the dictionary style LGBM prefers
    n_trees = fit_params["n_trees"]
    
    #Setting the numpy seed
    np.random.seed(fit_params["random_state"])

    #Optional Parameters
    if (isinstance(X_valid, pd.DataFrame) and isinstance(y_valid, pd.Series)):
        validation_data = lgb.Dataset(data=X_valid, label=y_valid) 
        early_stopping_rounds = fit_params["early_stopping_rounds"]
    elif ((X_valid == None) and (y_valid == None)):
        validation_data = None
        early_stopping_rounds = None
    else:
        ValueError("X_valid, y_valid need to be an instance of pd.DataFrame and pd.Series, respectively, if using validation sets. Otherwise, None for both parameters.")

    #LGBM throws warnings if these are not explicitly listed as arguments, instead of in the params
    del fit_params["n_trees"]
    del fit_params["early_stopping_rounds"]
    del fit_params["eval_metric"]
    del fit_params["maximize_metric"]

    #Training our model
    model = lgb.train(fit_params, train_data, valid_sets=validation_data, num_boost_round=n_trees, 
                      early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

    return model 



class lgbm():

    def __init__(self, nthread=1, set_nice=19, device_type="cpu", search_rounds=500,
                learning_rate_min=0.01, learning_rate_max=0.1, 
                min_gain_to_split_min=0.0001, min_gain_to_split_max=.5,
                max_depth_min=1, max_depth_max=5,
                min_data_in_leaf_min=10, min_data_in_leaf_max=50,
                bagging_frac_min=0.5, bagging_frac_max=1.0,
                col_sample_min=0.5, col_sample_max = 1.0,
                l1_min=.5, l1_max=50,
                l2_min=.5, l2_max=50,
                objective="rmse",
                eval_metric="rmse",
                maximize_metric=False,
                early_stop=25,
                verbose=False,
                nfold=5,
                max_bin=200,
                seed=6,
                fixed_parameters={}
                ):
        """
        Initializing the model with the hyperparameter space that we will be search over.
        We will use a Bayesian search for the hyperparameter space, using the hyperopt package.

        Inputs:
            nthreads: Number of cores to be used. Irrelevant if device_type=gpu
            set_nice: Assigning the priority for training, we don't want to bog down everyone
            learning_rate_min/learning_rate_max: The learning rate space we will consider
            min_gain_to_split_min/min_gain_to_split_max: Gain from a split required to continue 
            num_leaves_min/num_leaves_max: The number of splits space
            min_data_in_leaf_min/min_data_in_leaf_max: The min number of data points or sum of weights in a leaf
            bagging_frac_min/bagging_frac_max: bagging fraction space
            l1_min/l1_max: l1 penalty space
            l2_min/l2_max: l2 penalty space
            objective: The loss function we will be minimizing/maximizing
            eval_metric: evaluation metric for our validation and testing sets
                         built in functions are {ks, auc, rmse}
                         user-defined functions can be passed as well:
                            example: some_function(actual, predictions) returning a single score
            maximize_metric: hyperopt requires us to always minimize the metric so we need to multiple by -1 if True
            early_stop: early_stopping round for our validation set
            verbose: If True the metrics for training and validation sets will be printed for each gbdt
            nfold: number of folds for CV
            max_bin: Maximum number of times we can split up a single variable
            seed: Seed for reproducibility

        Parameter Definitions:
            https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        """

        #Making sure that the maximimize metric is correctly set
        if eval_metric in ["rmse", "auc", "ks", "mae"]:
            metric_direction_dict = {"rmse": False,
            						 "mae":False,
                                     "auc": True,
                                     "ks": True}

            maximize_metric = metric_direction_dict[eval_metric]

        #Saving off the parameter space
        self.nthread = nthread
        self.set_nice = set_nice
        self.device_type = device_type
        self.search_rounds = search_rounds
        self.learning_rate_min = learning_rate_min
        self.learning_rate_max = learning_rate_max
        self.min_gain_to_split_min = min_gain_to_split_min
        self.min_gain_to_split_max = min_gain_to_split_max
        self.max_depth_min = max_depth_min
        self.max_depth_max = max_depth_max
        self.min_data_in_leaf_min = min_data_in_leaf_min
        self.min_data_in_leaf_max = min_data_in_leaf_max
        self.bagging_frac_min = bagging_frac_min
        self.bagging_frac_max = bagging_frac_max
        self.col_sample_min = col_sample_min
        self.col_sample_max = col_sample_max
        self.l1_min = l1_min
        self.l1_max = l1_max
        self.l2_min = l2_min
        self.l2_max = l2_max
        self.objective = objective
        self.eval_metric = eval_metric
        self.maximize_metric = maximize_metric
        self.early_stop = early_stop
        self.verbose = verbose
        self.nfold = nfold
        self.max_bin = max_bin
        self.fixed_parameters = fixed_parameters
        
        self.seed = seed
        self.bagging_seed = seed + 1
        self.feature_fraction_seed = seed + 2
        self.data_random_seed = seed + 3
        
        #Training objects
        self.data = None
        self.feature_labels = None
        self.target_label = None
        self.search_space = None
        self.constant_params = None
        self.data_index = None
        self.indices_for_cv = None

        #Model Objects
        self.hyperparameters = None
        self.final_model_indices = None
        self.final_model_validation_perc = None
        self.model = None
        self.metrics = None
        self.trials = None


    def _training_data_error_check(self):
        """
        Checking the data provided for following points of failure:
            - target_label is in the features_labels
            - feature_labels are not in the data provided
            - target_label is not in the provided data
            - There is overlap between training, validation, and test sets. If the
              user has no provided indices for K-Folds, then the indices will be 
              generated.
        """

        #Type checking the arguments provided
        assert isinstance(self.data, pd.DataFrame), "Dataset provided should be of class pandas.DataFrame."

        if isinstance(self.feature_labels, str):
            self.feature_labels = list([self.feature_labels])

        #Checking to see that the feature_labels are all contained within the dataframe provided
        if not(set(self.feature_labels) <= set(self.data.columns)):
            raise ValueError("The following feature(s) are not in the dataframe provided:", set(self.feature_labels) - set(self.data.columns))

        #Checking that the target label is a column within the dataframe provided.
        if not(set([self.target_label]) <= set(self.data.columns)):
            raise ValueError("Target label is not in the columns in the dataframe provided.")

        #Checking that the user either provided all three lists of lists
        if self.indices_for_cv == None:
            #We will generate our own set of indices
            print("Using random indices for K-Folds.")
            
            self.indices_for_cv = list()
            for fold in range(0,self.nfold):
                #Creating the training, validation, testing indices for a single fold
                train_indices, test_indices = train_test_split(self.data_index, test_size=1/self.nfold)
                train_indices, valid_indices =  train_test_split(train_indices, test_size=1/self.nfold)
                
                #Creating the dictionary where the indices will be stored
                dictionary = dict()
                dictionary["train"] = train_indices 
                dictionary["valid"] = valid_indices
                dictionary["test"] = test_indices
                
                #Appending the list
                self.indices_for_cv.append(dictionary)
        
        elif isinstance(self.indices_for_cv, dict):
            #We need to check that there is no overlap of the indices
            print("Using user provided indices for K-Folds.")
            print("More than one fold for K-Folds cross-validation is advised.")

            #Setting the n_fold flag
            self.n_fold = 1
            
            if ((len(list(set(self.indices_for_cv["train"]) & set(self.indices_for_cv["valid"]))) > 0) or 
                (len(list(set(self.indices_for_cv["train"]) & set(self.indices_for_cv["test"]))) > 0) or 
                (len(list(set(self.indices_for_cv["valid"]) & set(self.indices_for_cv["test"]))) > 0)):
                raise ValueError("Indices for training, validing, testing have an overlap.") 
            
        elif isinstance(self.indices_for_cv, list):
            #We need to check that there is no overlap of the indices
            print("Using user provided indices for K-Folds.")

            #Setting the n_fold flag
            self.n_fold = len(self.indices_for_cv)
            
            for dictionary in self.indices_for_cv:
                if ((len(list(set(dictionary["train"]) & set(dictionary["valid"]))) > 0) or 
                (len(list(set(dictionary["train"]) & set(dictionary["test"]))) > 0) or 
                (len(list(set(dictionary["valid"]) & set(dictionary["test"]))) > 0)):
                    raise ValueError("Indices for training, validing, testing have an overlap.") 
        else:
            raise TypeError("Check documentation for acceptable types of arguments for indices_for_cv.")

    def _bayesian_hyperparameter_optimization(self, verbose=False):
        """
        Bayesian hyperparameter optimization

        Inputs:
            number_of_searches: Number of informed searches to perform
            verbose: Boolean for LGBM verbosity
            
        Output:
            best_hyperparameters: The selected hyperparameters
        """
        
        def _cross_validation(params, data, feature_labels, target_label, indices_for_cv):
            """
            K-Folds cross validation, we are using our own object instead of lgb.cv 
            because we want to use the same indices for training, validation, and
            testing for each parameter configuration.

            Input:
                params: hyperparameters
            Output:
                evaluation metric: sum of the test metric for all of the folds.

            """

            #Important variables
            test_metric = 0
            average_num_trees = 0
            nfolds=len(indices_for_cv)

            #Performing k folds cross validation
            for fold_dict in indices_for_cv:
                #Creating the fold
                X_train=data.loc[data.index.isin(fold_dict["train"]), feature_labels]
                y_train=data.loc[data.index.isin(fold_dict["train"]), target_label]
                X_valid=data.loc[data.index.isin(fold_dict["valid"]), feature_labels]
                y_valid=data.loc[data.index.isin(fold_dict["valid"]), target_label]
                X_test=data.loc[data.index.isin(fold_dict["test"])].copy()
                
                #Training the model
                fold_model = _fit_lgbm_model(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, params=params)

                #Updating the metric we are minimizing
                X_test["prediction"] = fold_model.predict(X_test[feature_labels])

                #Calculating the test metric
                if params["eval_metric"] == "ks":
                    test_metric = test_metric + ks_statistic(X_test[target_label], X_test["prediction"])/nfolds
                elif params["eval_metric"] == "auc":
                    test_metric = test_metric + auc(X_test[target_label], X_test["prediction"])/nfolds
                elif params["eval_metric"] == "rmse":
                    test_metric = test_metric + rmse(X_test[target_label], X_test["prediction"])/nfolds
                elif params["eval_metric"] == "mae":
                    test_metric = test_metric + mae(X_test[target_label], X_test["prediction"])/nfolds
                else:
                    try:
                        test_metric = test_metric + params["eval_metric"](X_test[target_label], X_test["prediction"])/nfolds
                    except:
                        TypeError("The eval_metric accepts {ks, auc, rmse} or a user-defined function accepting (actuals, predictions).")

                
                #Updating the average number of trees used for the model
                average_num_trees = average_num_trees + fold_model.num_trees()/nfolds
                
            #Cleaning up return values
            test_metric = round(test_metric,5)
            average_num_trees = int(average_num_trees)
            
            return test_metric, average_num_trees

        def _objective(params, constant_params, data, feature_labels, target_label, indices_for_cv):
            """
            Objective function for training a lgbm module with the given hyperparameters.

            """
            #Creating a single dictionary of the parameters
            params = {**params, **constant_params}
            
            #K-Folds cross validation
            test_metric, average_num_trees = _cross_validation(params=params, data=data, feature_labels=feature_labels, target_label=target_label, 
                                                            indices_for_cv=indices_for_cv)

            #Loss needs to be minimized
            if params["maximize_metric"]:
                loss = -1 * test_metric
            else:
                loss = test_metric

            #Dictionary required for the hyperopt package
            return {'loss': loss, 'n_trees': average_num_trees, 'status': STATUS_OK}    
        
        #Defining the search space
        self.search_space = {
                            "learning_rate":hp.uniform("learning_rate", self.learning_rate_min, self.learning_rate_max),
                            "min_gain_to_split":hp.uniform("min_gain_to_split", self.min_gain_to_split_min, self.min_gain_to_split_max),
                            "max_depth":hp.quniform("max_depth" ,self.max_depth_min, self.max_depth_max,1),
                            "min_data_in_leaf":hp.quniform("min_data_in_leaf" ,self.min_data_in_leaf_min, self.min_data_in_leaf_max,1),
                            "bagging_fraction":hp.quniform("bagging_fraction", self.bagging_frac_min, self.bagging_frac_max,0.1),
                            "feature_fraction":hp.quniform("feature_fraction", self.col_sample_min, self.col_sample_max,0.1),
                            "lambda_l1":hp.uniform("lambda_l1", self.l1_min, self.l1_max),
                            "lambda_l2":hp.uniform("lambda_l2", self.l2_min, self.l2_max)
                            }

        #Constant Parameters
        self.constant_params = {
                                "objective":self.objective,
                                "eval_metric":self.eval_metric,
                                "maximize_metric":self.maximize_metric,
                                "max_bin":self.max_bin,
                                "n_trees":10000,
                                "early_stopping_rounds": self.early_stop,
                                "bagging_freq":1,
                                "random_state":self.seed,
                                "bagging_seed":self.bagging_seed,
                                "feature_fraction_seed":self.feature_fraction_seed,
                                "data_random_seed":self.data_random_seed,
                                "verbosity":-1
                                }
        
        #Updating the constant parameters for user provided fixed variables
        self.constant_params = {**self.constant_params, **self.fixed_parameters}
        
        #Removing the fixed keys from the search space
        if len(self.fixed_parameters) > 0:
            for key in self.fixed_parameters:
                try:
                    del self.search_space[key]
                except:
                    pass
                
        #Setting the random state for hyperopt
        np.random.seed(self.seed)
        random_state = np.random.RandomState(self.seed)
        
        #Partial arguments
        fmin_objective = partial(_objective, constant_params = self.constant_params, data=self.data, feature_labels=self.feature_labels, 
                                 target_label = self.target_label, indices_for_cv=self.indices_for_cv)

        #Starting the hyperparameter search
        trials = Trials()
        optimized_params = hyperopt.fmin(fn = fmin_objective, space = self.search_space, algo=hyperopt.tpe.suggest, max_evals=self.search_rounds, trials=trials, rstate = random_state)

        #Grabbing the average number of trees used from the trials object
        all_results = pd.DataFrame(list(trials.results))
        average_num_trees = all_results.loc[all_results.loss.min() == all_results.loss, "n_trees"].values[0]
        self.constant_params["n_trees"] = average_num_trees

        #Setting our hyperparameters that will be used in the final model
        self.hyperparameters = {**optimized_params, **self.constant_params} 
        self.trials = trials


    def hyperparameter_search(self, data, feature_labels, target_label, indices_for_cv=None):
        """
        Performing the hyperparameter search

        The user can choose to provide the train, validation, and test sets for train
        or it can generate the train, validation, and test sets.
        
        Inputs:
            dataset: A pandas dataframe containing the feature space and the target.
            feature_labels: The feature space we will be using 
            target_label: The target we will be predicting
            indices_for_cv: A list of dictionaries for cross validation. Example:
                            indices_for_cv = [{
                                             "train": [1,4,5],
                                             "valid": [2],
                                             "test": [3]
                                             },
                                             {
                                             "train": [2,3,4],
                                             "valid": [1],
                                             "test": [5]
                                             },
                                             ...]
            
        """

        #Saving off the arguments
        self.data = data
        self.data_index = np.array(data.index)
        self.feature_labels = feature_labels
        self.target_label = target_label
        self.indices_for_cv = indices_for_cv
        
        #Error checking the dataset and generating indices for k-folds
        self._training_data_error_check()
        
        #Bayesian hyperparamter search
        self._bayesian_hyperparameter_optimization()
       
    def train(self):
        """
        Training our lightgbm module with the hyperparameter selected
        using Bayesian Optimization.
        
        Inputs:
            final_model_indices: a dictionary containing the training and 
                                 validation indices. Example:
                                     {"train": [1,4,5,6],
                                      "valid": [2,3]}
        Output:
            None
        """
        
        #Make sure that we have done the hyperporameter search
        assert self.hyperparameters != None, "Perform the hyperparameter search prior to training the final model."

        #Training the final model
        X_train = self.data[self.feature_labels]
        y_train = self.data[self.target_label]
        
        self.model = _fit_lgbm_model(params=self.hyperparameters, X_train=X_train, y_train=y_train)
        
        #Calculating the feature importance
        self.feature_importance = pd.DataFrame(data={"feature": self.model.feature_name(),
                                                     "feature_importance": self.model.feature_importance(importance_type="gain")})
        self.feature_importance["feature_importance"] = self.feature_importance["feature_importance"]/sum(self.feature_importance["feature_importance"])
        self.feature_importance = self.feature_importance.sort_values(by="feature_importance",ascending=False).reset_index(drop=True)
                                                        
    def predict(self, X, pred_contrib=False):
        """
        Making predictions for the given dataset
        
        Inputs:
            X: Dataset we will be making predictions for
            pred_contribution: Returnings the prediction, as well as the 
                               contribution for each of variables. See the 
                               LightGBM documentation for more information
        """
        #Making sure that the model has been train before we try to use
        assert self.model != None, "Model needs to be trained before trying to make predictions."
        
        #Making the predictions
        if pred_contrib:
            contributions_df = pd.DataFrame(data=self.model.predict(X[self.feature_labels], pred_contrib=pred_contrib))
            contributions_df.columns = self.model.feature_name() + ["average_prediction"]
            return contributions_df
        else:
            return self.model.predict(X[self.feature_labels])
        
    def adverse_action(self, X):
        assert self.model != None, "Model needs to be trained before trying to make predictions."
        contributions_series = pd.Series(data=self.model.predict(X[self.feature_labels], pred_contrib=True)[0], index=self.model.feature_name() + ["average_prediction"])
        contributions_series = contributions_series.drop(labels="average_prediction")
        contributions_series = contributions_series.sort_values(ascending=False)
        reasons_for_prediction = pd.DataFrame(data = {"Impact_on_Prediction":contributions_series.values, "Feature":contributions_series.index})
        
        inputs = pd.DataFrame(data={"Feature":self.feature_labels,"Feature_Values":list(X[self.feature_labels])})
        reasons_for_prediction = reasons_for_prediction.merge(inputs, on=["Feature"], how="left")
        return reasons_for_prediction
            
        
        
        
        
        
        
        
        
                                                         