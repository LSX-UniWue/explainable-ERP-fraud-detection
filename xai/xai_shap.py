import os
import warnings
import numpy as np
import pandas as pd
import shap

import dask.array as da
from dask_ml.wrappers import ParallelPostFit


def shap_explain(detector, data, baseline=None, out_template=None, **shap_kwargs):
    """
    SHAP for the tabular regression task

    SHAP behavior on Regression:
    shap_values[0].sum(1)      + explainer.expected_value - model.predict(X, raw_score=True) ~ 0
    attribution sum per sample + expected value           - output                           ~ 0
    """

    # Setting up SHAP
    # Background dataset for "default" values when "removing"/perturbing features:
    # Recommends cluster centers from training dataset or zero-vector, depending on what "missing" means in data context
    if baseline is None:  # default is 0 for missing feature
        baseline = np.zeros(data.shape[1]).reshape(1, data.shape[1])
    elif len(baseline.shape) > 1 and baseline.shape[0] > 100:
        baseline = shap.kmeans(baseline, k=20)

    # function producing target outputs
    if isinstance(detector, ParallelPostFit):  # trick for multiprocessing single core algorithms with dask
        def predict_fn(X):
            data = da.from_array(X, chunks=(100, -1))
            return detector.predict(data).compute()
    else:
        predict_fn = detector.score_samples

    shap_xai = shap.KernelExplainer(predict_fn, baseline, **shap_kwargs)

    # get shap values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = shap_xai.shap_values(data)

    print(f"SHAP expected_value: {shap_xai.expected_value}")

    if out_template:
        if isinstance(data, pd.Series):
            out_df = pd.DataFrame(shap_values, columns=[data.name], index=data.index.values).T
            out_df['expected_value'] = shap_xai.expected_value
        else:
            out_df = pd.DataFrame(shap_values, index=data.index, columns=data.columns)
        if os.path.exists(out_template.format('shap')):
            old_df = pd.read_csv(out_template.format('shap'), header=0, index_col=0)
            out_df.columns = out_df.columns.astype(str)  # read_csv reads column names as str by default
            out_df = pd.concat([old_df, out_df])
        out_df.to_csv(out_template.format('shap'))

    return shap_xai, shap_values


def anomaly_shap_values(X_anomalous, X_benign, detector, out_template):
    """
    Generates Shap explanations with different background datasets depending on background, saves to out_template
    :param X_anomalous:     pd.DataFrame including data to explain
    :param X_benign:        pd.DataFrame including benign data to sample background data from
    :param detector:        Anomaly detector with score_samples function
    :param out_template:    Str path to output .csv file, containing 1 format slot {} for XAI name
    """
    print("Calculating SHAP values...", flush=True)
    # SHAP default kmeans sampling from X_train
    _, shap_expl = shap_explain(detector=detector,
                                data=X_anomalous,
                                baseline=X_benign,
                                out_template=out_template)
