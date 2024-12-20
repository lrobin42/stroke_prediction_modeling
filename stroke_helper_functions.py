import polars as pl
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from skimpy import skim
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from xgboost import XGBRFClassifier
from scipy.stats import pointbiserialr, fisher_exact
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    recall_score,
    precision_score,
    fbeta_score,
)
import matplotlib.pyplot as plt
import pickle
from fastapi import FastAPI
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import shap
from sklearn import tree
import plotly.io as pio
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.metrics import make_scorer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import itertools
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from pyod.models.knn import KNN
from sklearn.tree import DecisionTreeClassifier


def false_negative_error_rate(y_true, y_predict)-> float:
    """Inputs arrays of y and predicted y values to calculate the false negative rate of model predictions

    Args:
        y_true (numpy array or data series): actual dependent variable values from the dataset
        y_predict (numpy array or data series): predicted dependent variables created by the classifier

    Returns:
        : false_negative_rate within model
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()

    false_negative_rate = fn / (tn + fp + fn + tp)
    return false_negative_rate

def make_subplot(df, figure, feature, position):
    """Makes bar subplot for row and column of figure specified

    Args:
        figure (plotly go): Plotly graph_objects figure
        feature (string): the name of the column within pandas df to make plot of
        position (list): list of integers in [row,column] format for specifying where in figure to plot graph
        labels (list, optional): Title, xlabel, and ylabel for subplots. Defaults to ['',None,None].
    """

    tallies = df[feature].sort_values(ascending=True).value_counts()
    figure.add_trace(
        go.Bar(
            x=tallies.index,
            y=tallies.values,
            name="",
            marker=dict(
                color=[
                    "rgb(191,237,204)",
                    "rgb(76,145,151)",
                    "rgb(33,92,113)",
                    "rgb(22,70,96)",
                ]
            ),
            hovertemplate="%{x} : %{y}",
            text=tallies.values,
        ),
        row=position[0],
        col=position[1],
    )
    figure.update_layout(bargap=0.2)

def create_encoder_mapping(df, feature)->dict[str,int]:
    """Creates dictionary for mapping to encode categorical features

    Args:
        df (polars dataframe): dataframe of features
        feature (string): name of feature of interest

    Returns:
        encoding_key: dictionary of feature values and numbers for encoding
    """
    df = (
        df.group_by(feature)
        .agg(pl.len().alias("values"))
        .sort("values", descending=True)
    )

    options = df[feature].to_list()

    numbers_to_encode = list(range(0, len(options)))
    encoding_key = {options[i]: numbers_to_encode[i] for i in range(len(options))}

    if df[feature].str.contains("Yes").to_list()[0] == True:
        encoding_key = {"Yes": 1, "No": 0}

    return encoding_key

def encode_feature(df, feature, encoding_key)-> -> pl.DataFrame:
    """Encode features using supplied encoding key

    Args:
        df (polars): Dataframe to be modified
        feature (string): feature to be encoded
        encoding_key (dict): dictionary of values and numerical codes

    Returns:
        df: input dataframe with feature replaced by numerical values
    """
    df = df.with_columns(
        df.select(pl.col(feature).replace(encoding_key)).cast({feature: pl.Int64})
    )

    return df

def impute_missing_values(df, feature, method, format) -> pd.DataFrame:
    """Impute missing values with sklearn simple imputer

    Args:
        df (polars): dataframe
        feature (string): feature to be imputed
        method (string): specified strategy parameter for SimpleImputer
        format (string): specifies value used for missing value in df

    Returns:
        imputed_df (polars dataframe): dataframe with imputed values for feature given
    """
    # save columns
    columns = df.columns
    # convert column to numpy_array
    array = df.to_numpy()

    # create simple imputer instance
    imputer = SimpleImputer(strategy=method, missing_values=format)

    # impute missing values
    imputed_array = imputer.fit_transform(array)

    # Convert array back to polars
    imputed_df = pl.DataFrame(imputed_array)

    # Overwrite columns
    imputed_df.columns = columns

    # output modified dataframe
    return imputed_df

def create_heatmap(df, reversed=True):
    """Creates a heatmap of correlations

    Args:
        df (dataframe): correlation matrix
        reversed (bool, optional): _description_. Defaults to True.
    """
    cmap = sns.color_palette("crest", as_cmap=True).reversed(
        sns.color_palette("dark:#5A9_r", as_cmap=True)
    )
    if reversed == False:
        cmap = sns.color_palette("crest", as_cmap=True)
    sns.heatmap(
        df,
        annot=True,
        cmap=cmap,
    )
    plt.show()
def int_range(start, end) -> np.ndarray[np.int, np.dtype[np.int]]:
    """Generate np.linspace range for limits given, such that all inclusive consecutive integers are included

    Args:
        start (int): lower limit of range
        end (int): upper limit of range

    Returns:
        numpy array of range: integer range of values
    """
    return np.linspace(start, end, len(range(start, end + 1)), dtype=int)
def calculate_model_statistics(y_true, y_predict, beta=3.0, title="statistics"):
    """Uses actual y and predicted y values to return a dataframe of accuracy, precision, recall, and f-beta values as well as false negative and false posititive rates for a given classifier

    Args:
        y_true (numpy array or data series): dependent variable values from the dataset
        y_predict (_type_): dependent variable values arising from model
        beta (float, optional): Beta value to determine weighting between precision and recall in the f-beta score.Defaults to beta value set in global scope of this notebook.
        title (str, optional): _description_. Defaults to "statistics".

    Returns:
        model_statistics: pandas dataframe of statistics
    """

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()

    # calculate statistics from confusion matrix
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f_beta = fbeta_score(y_true, y_predict, beta=beta)
    false_negative_rate = fn / (tn + fp + fn + tp)
    false_positive_rate = fp / (tn + fp + fn + tp)

    return pd.DataFrame(
        data={
            title: [
                f_beta,
                recall,
                precision,
                accuracy,
                false_negative_rate,
                false_positive_rate,
            ]
        },
        index=[
            "f_beta",
            "recall (sensitivty)",
            "precision (specificity)",
            "accuracy",
            "false_negative_rate",
            "false_positive_rate",
        ],
    )
def resplit_data(X, Y, test_size=0.3) -> NDArray | DataFrame:
    """Re-execute stratified train-test-split

    Args:
        X (df): features
        Y (df): labels
        test_size (float, optional): Specify proportion of test_set. Defaults to 0.3.

    Returns: 
        x_train (df): x training set
        x_test (df): x testing set
        y_train (df): y training set 
        y_test (df): y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=15
    )
    return x_train, x_test, y_train, y_test

def conduct_grid_search_tuning(
    model, grid, x_train, y_train, refit, scoring=make_scorer(fbeta_score, beta=2), cv=5
):
    """Conducts gridsearch for specified model and hyperparameter settings

    Args:
        model (string): string specifying model to test, must be 'knn', 'logistic_regression','decision_tree', or 'random_forest'
        grid (dictionary): grid of lists specifying options for hyperparameters to tune
        xy (list): x and y for model fitting, should be in [x_train,y_train] format
        scoring(string/callable): string defines scoring method to be used within grid search
    """

    grid_search = GridSearchCV(
        model, grid, cv=cv, scoring=scoring, refit=refit, n_jobs=-1
    )
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_

    return best_params  # , grid_search

def segment_df_by_gender(df, gender_series)-> pl.DataFrame:
    """Segments polars df by gender, returns male and female only versions of data
    
    Args: 
        df (polars): dataframe to be segmented 
        gender_series (polars series): gender series to be added to dataset if the column is not present in the dataframe
    
    Returns:
        male_df: polars dataframe of values corresponding to male patients
        female_df: polars dataframe of values corresponding to female patients
    """
    # Check if there is a gender column in the data. If not, add gender to dataframe
    try:
        df["gender"]
    except:
        df.insert_column(0, gender_series)

    # Filter on male and female
    male_df = df.filter(pl.col("gender") == 1).drop("gender")
    female_df = df.filter(pl.col("gender") == 0).drop("gender")

    # Return male df and female df
    return male_df, female_df

def polars_value_counts(df,feature) -> pl.DataFrame:
    return df.group_by(feature).agg(pl.len())

def create_contingency_table(df, column1, column2):
    """Calculates contingency matrix for two variables within binary_df

    Args:
        column1 (str): first variable
        column2 (str): second variable

    Returns:
        contingency table in pandas dataframe format
    """
    if type(df) == pl.dataframe.frame.DataFrame:
        df = df.to_pandas()
    matrix = pd.crosstab(df[column1], df[column2], margins=False)
    return matrix

def fishers_exact_test(matrix) -> tuple[_T_co, _T_co]:
    """Function conducts fisher's exact test on two binary variables via the contingency table between them

    Args:
        matrix (pandas df): a contingency table

    Returns:
        observed_odds_ratio: the observed odds_ratio in the dataset, which is assumed to be 1 under the null hypothesis
        p_value: p_value of seeing an equally extreme value under null hypothesis
    """
    observed_odds_ratio, p_value = fisher_exact(matrix)
    return observed_odds_ratio, p_value

def restrict_grid_search_table(grid_search_results_df, metrics) -> DataFrame:
    """Function takes grid_search printout and filters down to metrics-related columns and metrics-ranges
    Args:
        grid_search_results_df (df): complete output of grid search
        metrics (list): list of strings matching metrics of interest
        grid_search_ranges (df): dataframe with (min,max) ranges for each metric in metrics list

    Returns:
        dataframe: df with metrics and metrics ranges
    """
    regex = "|".join(metrics)
    columns = grid_search_results_df.filter(regex=regex).columns
    df = grid_search_results_df[columns]
    return df


def test_score_range(df) -> list:
    return df.apply(
        lambda row: [np.round(row.min(), 3), np.round(row.max(), 3)], axis=1
    )


def calculate_grid_search_ranges(df, metrics) -> pd.DataFrame:
    """Calculates range of test_scores for all metrics specified

    Args:
        metrics (list): list of strings denoting metrics

    Returns:
        output: dataframe of relevant columns from overall gridsearch output
    """
    output = pd.DataFrame()
    for metric in metrics:
        pattern = "split\\d_test_" + metric
        data = df.filter(regex=pattern)
        column_name = metric + "_range"
        output[column_name] = test_score_range(data)
    return output

def lgbm_feature_importances(lgbm_model, scaled=False):
    """Calculate both gain and split importances of specific lightgbm classifier

    Args:
        lgbm_model (_type_): _description_
        scaled (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    gain_importances = lgbm_model.booster_.feature_importance(importance_type="gain")
    split_importances = lgbm_model.booster_.feature_importance(importance_type="split")

    if scaled:
        gain_importances = (
            lgbm_model.booster_.feature_importance(importance_type="gain")
            / lgbm_model.n_estimators_
        )
        split_importances = (
            lgbm_model.booster_.feature_importance(importance_type="split")
            / lgbm_model.n_estimators_
        )

    return gain_importances, split_importances

def check_grid_permutations(hyperparameter_grid) -> int:
    """Quick function to calculate the number of model permutations created by a hyperparameter grid, to help ring-fence runtimes in GridSearchCV

    Args:
        hyperparameter_grid (dictionary): dictionary of arrays or lists specifying hyperparameter configurations

    Returns:
        combinations: integer representing number of distinct model specifications in the hyperparameter grid
    """
    combinations = list()
    for i in hyperparameter_grid.keys():
        combinations.append(len(hyperparameter_grid[i]))
    return np.prod(combinations, dtype=int)

def create_fishers_array(binary_df) -> pd.DataFrame:
    """Creates array of fisher's exact test p-values
    
    Args: 
        binary_df: dataframe of binary features
    
    Returns: 
        array: Pandas dataframe of p-values for for all pairwise fisher's exact tests on variables in the array
    """
    variables = binary_df.columns

    array = pd.DataFrame(
        np.zeros((len(variables), len(variables)), dtype=int),
        columns=variables,
        index=variables,
    )

    for column in variables:
        array[column] = [
            fishers_exact_test(create_contingency_table(binary_df, column, x))[1]
            for x in variables
        ]
    return array

def polars_crosstab(df, col_a,col_b) -> pl.DataFrame: 
    crosstab = df.pivot(values=col_a, index=col_b, columns=col_a, aggregate_function="count").fill_null(0)
    return crosstab


def calculate_chi2(df, col_a, col_b) -> float:   
    crosstab = polars_crosstab(df, col_a, col_b)
    stats, p_value, dof, array = chi2_contingency(crosstab)
    return p_value

def create_chi2_array(binary_df) -> pd.DataFrame:
    """Creates array of fisher's exact test p-values
    
    Args: 
        binary_df: dataframe of binary features
    
    Returns: 
        array: Pandas dataframe of p-values for for all pairwise fisher's exact tests on variables in the array
    """
    variables = binary_df.columns

    array = pd.DataFrame(
        np.zeros((len(variables), len(variables)), dtype=int),
        columns=variables,
        index=variables,
    )

    for column in variables:
        array[column] = [
            calculate_chi2(binary_df,x, column)
            for x in variables
        ]
    return array
