
import numpy as np
import pandas as pd
from abc import ABC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from category_encoders.one_hot import OneHotEncoder

from data.nan_discretizise import NanDiscretizer


class Preprocessor(ABC):
    """Generic Preprocessing object for creating numerical and categorical data preprocessors"""

    def __init__(self):
        self.encoder = None
        self.is_fitted = False

    def fit(self, data):
        pass

    def transform(self, data):
        pass


class CategoricalOneHotPreprocessor(Preprocessor):

    def __init__(self):
        super(CategoricalOneHotPreprocessor, self).__init__()

    def fit(self, data):
        data = data.fillna('nan').astype('str')  # need to replace nan values since nan != nan for oh_encoder
        self.encoder = OneHotEncoder(handle_missing="value", use_cat_names=True)
        self.encoder.fit(data.astype('str'))
        self.is_fitted = True

    def transform(self, data):
        return self.encoder.transform(data.astype('str'))


class NumericalZscorePreprocessor(Preprocessor):
    """
    Applies Zscore scaling to each numerical column, optionally with additional binary column marking empty values.
    :param nan_bucket:      Additional binary column marking nan/0 values for each numerical column
    :param numeric_last:    Orders numeric attributes to the back and onehot-nan-buckets to the front if set to True
    """
    def __init__(self, nan_bucket=False, numeric_last=False, **kwargs):
        super(NumericalZscorePreprocessor, self).__init__()
        self.nan_bucket = nan_bucket
        self.numeric_last = numeric_last

    def fit(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        self.encoder = StandardScaler()
        self.encoder.fit(data)
        self.is_fitted = True

    def transform(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        data_fit = pd.DataFrame(self.encoder.transform(data), columns=data.columns, index=data.index)
        if self.nan_bucket:
            empty = (data == 0).astype(int)
            data_fit = data_fit.join(empty.rename({col_name: col_name + '_0' for col_name in empty.columns}, axis=1))
        return data_fit


class NumericalMinMaxPreprocessor(Preprocessor):
    """
    Applies min-max scaling to each numerical column, optionally with additional binary column marking empty values.
    :param nan_bucket:      Additional binary column marking nan/0 values for each numerical column
    :param numeric_last:    Orders numeric attributes to the back and onehot-nan-buckets to the front if set to True
    """
    def __init__(self, nan_bucket=False, numeric_last=False, **kwargs):
        super(NumericalMinMaxPreprocessor, self).__init__()
        self.nan_bucket = nan_bucket
        self.numeric_last = numeric_last

    def fit(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        self.encoder = MinMaxScaler()
        self.encoder.fit(data)
        self.is_fitted = True

    def transform(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        data_fit = pd.DataFrame(self.encoder.transform(data), columns=data.columns, index=data.index)
        if self.nan_bucket:
            empty = (data == 0).astype(int)
            data_fit = data_fit.join(empty.rename({col_name: col_name + '_0' for col_name in empty.columns}, axis=1))
        return data_fit


class NumericalQuantizationPreprocessor(Preprocessor):
    """
    Applies Quantization to each numerical column, converting each column into multiple buckets,
    with an extra bucket for nan/0.
    Additionally makes left and rightmost buckets 1% buckets for detecting outliers.
    """
    def __init__(self, n_buckets=5, encode='onehot', **kwargs):
        super(NumericalQuantizationPreprocessor, self).__init__()
        self.n_buckets = n_buckets
        self.encode = encode

    def fit(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        self.encoder = NanDiscretizer(n_bins=self.n_buckets, encode=self.encode, strategy='quantile_outlier')
        self.encoder.fit(data)

    def transform(self, data):
        data = data.fillna(0)  # Treat 0 and nan in numeric cols as missing vals
        return self.encoder.transform(data)
