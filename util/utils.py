import pandas as pd
from typing import Union
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import is_classifier, is_regressor
import torch


def manipulate_testdata(xtrain: pd.DataFrame, 
                        xtest: pd.DataFrame, 
                        model: Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator,
                               torch.nn.Module],
                        variable: str):
    """
    Manipulate the given variable column in test data based on values of that variable in train data.

    Parameters
    ----------
    xtrain : pd.DataFrame
            A dataframe including train data.
    xtest : pd.DataFrame
            A dataframe including test data.
    model : Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor.
    variable: str 
            Name of variable.

    Returns
    -------
    pd.DataFrame
            The manipulated data.
    """
    # create xtest_rm
    xtest_rm = xtest.copy()
    if isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
        # specific settings for CatBoost models
        cat_indices = model.get_cat_feature_indices()
        # replace variable with mode or mean based on its type
        if xtrain.columns.get_loc(variable) not in cat_indices:
            mean_value = xtrain[variable].mean()
            xtest_rm[variable] = mean_value
        else:
            mode_value = xtrain[variable].mode()[0]
            xtest_rm[variable] = mode_value
        
    elif isinstance(model, (BaseEstimator, XGBClassifier, XGBRegressor)):
        # specific settings for sklearn and xgboost models
        if isinstance(xtrain[variable].dtype, pd.CategoricalDtype):
            mode_value = xtrain[variable].mode()[0]
            xtest_rm[variable] = mode_value
        else:
            mean_value = xtrain[variable].mean()
            xtest_rm[variable] = mean_value

    elif isinstance(model, torch.nn.Module):
        # specific settings for torch models
        if isinstance(xtrain[variable].dtype, pd.CategoricalDtype):
            mode_value = xtrain[variable].mode()[0]
            xtest_rm[variable] = mode_value
        else:
            mean_value = xtrain[variable].mean()
            xtest_rm[variable] = mean_value
    else:
        raise ValueError("Unsupported model type")
    return xtest_rm


def convert_to_dataframe(*args):
    """
    Convert inputs to DataFrames.

    Parameters
    ----------
    Args:  *args
            A variable number of input objects that can be converted into Pandas DataFrames (e.g., lists, dictionaries, numpy arrays).

    Returns
    -------
    list of pd.DataFrame
            A list of Pandas DataFrames created from the input objects.    
    """
    return [pd.DataFrame(arg).reset_index(drop=True) for arg in args]


def validate_variables(variables: list, xtrain: pd.DataFrame):
    """
    Check if variables are valid and exist in the train dataset.

    Parameters
    ----------
    variables: list 
            List of variables.
    xtrain : pd.DataFrame
            A dataframe including train data.

    Raises
    -------
    ValueError
            If variables is not a list or if any variable does not exist in xtrain.
    """
    if not isinstance(variables, list):
        raise ValueError("Variables input must be a list")
    for var in variables:
        if var not in xtrain.columns:
            raise ValueError(f"{var} is not in the variables")


def check_nan(*dataframes):
    """
    Check if any of the provided DataFrames contain missing values.

    Parameters
    ----------
    *dataframes : pd.DataFrame
        A variable number of DataFrame objects to check for NaN values.

    Raises
    ------
    ValueError
        If any DataFrame contains missing (NaN) values.
    TypeError
        If any input is not a Pandas DataFrame.
    """

    for i, df in enumerate(dataframes, start=1):
        if df.isna().sum().sum() > 0:
            raise ValueError(f"DataFrame {i} contains missing values.")


def find_yhat(model: Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator,
              torch.nn.Module],
              xtest: pd.DataFrame):
    """
    Find predicted values for the manipulated data.

    Parameters
    ----------
    xtest : pd.DataFrame
            A dataframe including test data.
    model : Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor.

    Returns
    -------
    float
            The yhat value.
    """
    if is_classifier(model):
        yhat = [x[1] for x in model.predict_proba(xtest)]
    elif is_regressor(model):
        yhat = model.predict(xtest)
    elif isinstance(model, torch.nn.Module):
        xtest_tensor = torch.tensor(xtest.values, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            yhat = model(xtest_tensor)
        if yhat.shape[1] == 2:  # binary classification
            yhat = yhat[:, 1].numpy()
        else:
            yhat = yhat.numpy()
    return yhat