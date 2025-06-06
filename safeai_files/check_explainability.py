from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd
from sklearn.base import BaseEstimator
import torch
from typing import Union
from safeai_files.utils import check_nan, convert_to_dataframe, find_yhat, manipulate_testdata, validate_variables
from xgboost import XGBClassifier, XGBRegressor

from safeai_files.core import rga

def compute_rge_values(xtrain: pd.DataFrame, 
                xtest: pd.DataFrame,
                yhat: list,
                model: Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator,
                torch.nn.Module],
                variables: list, 
                group: bool = False):
    """
    Helper function to compute the RGE values for given variables or groups of variables.

    Parameters
    ----------
    xtrain : pd.DataFrame
            A dataframe including train data.
    xtest : pd.DataFrame
            A dataframe including test data.
    yhat : list
            A list of predicted values.
    model : Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor. 
    variables : list
            A list of variables.
    group : bool
            If True, calculate RGE for the group of variables as a whole; otherwise, calculate for each variable.

    Returns
    -------
    pd.DataFrame
            The RGE values for each variable or for the group.
    """
    # Convert inputs to DataFrames and concatenate them
    xtrain, xtest, yhat = convert_to_dataframe(xtrain, xtest, yhat)
    # check for missing values
    check_nan(xtrain, xtest, yhat)
    # variables should be a list
    validate_variables(variables, xtrain)
    # find RGEs
    if group:
        # Apply manipulate_testdata iteratively for each variable in the group
        for variable in variables:
            xtest = manipulate_testdata(xtrain, xtest, model, variable)
        
        # Calculate yhat after manipulating all variables in the group
        yhat_rm = find_yhat(model, xtest)
        
        # Calculate a single RGE for the entire group except these variables
        rge = rga(yhat, yhat_rm)
        return pd.DataFrame([rge], index=[str(variables)], columns=["RGE"])

    else:
        # Calculate RGE for each variable individually
        rge_list = []
        for variable in variables:
            xtest_rm = manipulate_testdata(xtrain, xtest, model, variable)
            yhat_rm = find_yhat(model, xtest_rm)
            rge_list.append(1 - (rga(yhat, yhat_rm)))
        
        return pd.DataFrame(rge_list, index=variables, columns=["RGE"]).sort_values(by="RGE", ascending=False)

