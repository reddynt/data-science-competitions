# imports

import json
import warnings
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
)

N_SPLITS = 7

ordinal_cols_order = {
    "MSSubClass": [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Gtl", "Mod", "Sev"],
    "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"],
    "ExterCond": ["Ex", "Gd", "TA", "Fa", "Po"],
    "HeatingQC": ["Ex", "Gd", "TA", "Fa", "Po"],
    "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po"],
    "Functional": ["Typ", "Min1", "Min2", "Maj1", "Maj2", "Sev", "Sal"],
    "FireplaceQu": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
    "GarageFinish": ["Fin", "RFn", "Unf", "NA"],
    "GarageCond": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
    "GarageQual": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
    "BsmtExposure": ["Gd", "Av", "Mn", "No", "NA"],
    "BsmtFinType2": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
    "BsmtFinType1": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
    "BsmtCond": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
    "BsmtQual": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
    "PoolQC": ["Ex", "Gd", "TA", "Fa", "NA"],
    "Fence": ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"],
}


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def get_feature_names_out(self):
        return self.cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.cols]


class TransformDate(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["age_when_sold"] = X["YrSold"] - X["YearBuilt"]
        X["is_remodeled"] = X["YearBuilt"] != X["YearRemodAdd"]
        X["garage_present"] = X["GarageYrBlt"].apply(lambda x: True if x else False)
        X.loc[:, "GarageYrBlt"].fillna(X.YearBuilt, inplace=True)
        return X


class OrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ord_encoders = dict()

    def get_feature_names_out(self):
        return np.array(list(self.ord_encoders.keys()))

    def fit(self, X, y=None):
        for col in X.columns:
            categories_order = ordinal_cols_order.get(col, None)
            categories_order = (
                "auto" if categories_order is None else [categories_order]
            )
            encoder = OrdinalEncoder(
                categories=categories_order,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            encoder.fit(X[[col]])
            self.ord_encoders[col] = encoder
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            encoder = self.ord_encoders[col]
            X.loc[:, col] = encoder.transform(X[[col]])
        return X


def get_paths():
    if Path("/kaggle/input").exists():
        dataset_path = Path("/kaggle/input/house-prices-advanced-regression-techniques")
        result_path = Path("/kaggle/working/")
    else:
        dataset_path = Path("./dataset/")
        result_path = Path("./results")
        result_path.mkdir(parents=True, exist_ok=True)

    return dataset_path, result_path


def list_files(path):
    if not isinstance(path, Path):
        path = Path(path)
    print([_.name for _ in path.iterdir()])


def config_output(n_rows=1000, n_cols=1000, ignore_warnings=True):
    pd.options.display.max_rows = n_rows
    pd.options.display.max_columns = n_cols
    plt.style.use("fivethirtyeight")
    print(f"setting no. of rows displayed: {n_rows}")
    print(f"setting no. of columns displayed: {n_cols}")
    if ignore_warnings:
        print(f"setting warnings to be ignored.")
        warnings.filterwarnings("ignore")
    print("\n")


Param = namedtuple("Param", ["name", "dtype", "param_range"])

rfr_params = {
    "n_estimators": Param("n_estimators", "integer", (100, 1000)),
    "criterion": Param("criterion", "category", ("squared_error", "absolute_error")),
    "min_samples_split": Param("min_samples_split", "integer", (2, 5)),
    "max_depth": Param("max_depth", "integer", (2, 11)),
    "max_features": Param("max_features", "category", ("auto", "sqrt", "log2")),
    "bootstrap": Param("boostrap", "category", (True, False)),
}


class Blending:

    def __init__(self, name, train_df, test_df, target_col, preprocess_pipe, fold_col=None):
        self.name = name
        self.train_df = train_df
        self.test_df = test_df
        self.fold_col = fold_col
        self.target_col = target_col
        self.preprocess_pipe = preprocess_pipe
        
    def objective(self, trial):
        
        def compose_trial(param):
            if param.dtype == "integer":
                return trial.suggest_int(param.name, *param.param_range)
            elif param.dtype == "float":
                return trial.suggest_float(param.name, *param.param_range)
            elif param.dtype == "category":
                return trial.suggest_categorical(param.name, param.param_range)

        fold_scores = []
        params = dict()

        for key, value in rfr_params.items():
            params[key] = compose_trial(value)
        
        # iterate through different folds
        for fold in range(N_SPLITS):
            # form xtrain & xvalid datasets
            xtrain = self.train_df.loc[self.train_df[self.fold_col] != fold, :]
            xvalid = self.train_df.loc[self.train_df[self.fold_col] == fold, :]
            ytrain = self.train_df.loc[self.train_df[self.fold_col] != fold, self.target_col]
            yvalid = self.train_df.loc[self.train_df[self.fold_col] == fold, self.target_col]

            xtrain = self.preprocess_pipe.fit_transform(xtrain)
            xvalid = self.preprocess_pipe.transform(xvalid)
            
            ytrain = np.log1p(ytrain)

            model = RandomForestRegressor(**params, random_state=13)

            model.fit(xtrain, ytrain)

            ypreds = model.predict(xvalid)
            ypreds = [np.expm1(_) for _ in ypreds]

            fold_rmse = mean_squared_log_error(yvalid, ypreds, squared=False)
            fold_scores.append(fold_rmse)

        return np.mean(fold_scores)
        

    def optimize(self, direction="maximize", n_trials=1, timeout=600):
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(self.objective, n_trials=n_trials)


if __name__ == "__main__":
    pass
