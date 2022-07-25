import os

import numpy as np
import pandas as pd
from data.preprocessing import CategoricalOneHotPreprocessor, NumericalZscorePreprocessor, \
    NumericalQuantizationPreprocessor, NumericalMinMaxPreprocessor


def get_column_dtypes(source_folder, language):
    """Two lists of column headers for categorical and numerical columns. Needs to be updated on data changes!"""
    column_info = pd.read_csv(os.path.join(source_folder, 'ex2/column_information.csv'), index_col=0,
                              header=0).T
    cat_cols = column_info[column_info['cat'].astype(float) == 1][language].values
    num_cols = column_info[column_info['num'].astype(float) == 1][language].values
    return cat_cols, num_cols


def load_and_preprocess(source_folder,
                        dataset_name,
                        numeric_preprocessing,
                        categorical_preprocessing='onehot',
                        language='ger',
                        keep_notes=False,
                        keep_index=False,
                        keep_label=False,
                        keep_original_numeric=False,
                        **kwargs):
    """
    Loads data with different numerical preprocessing techniques.
    :param source_folder:               String Path to general data folder
    :param dataset_name:                String name of specific dataset
    :param numeric_preprocessing:       One of ['zscore', 'buckets', 'minmax', None]
    :param categorical_preprocessing:   One of ['onehot', None]
    :param language:                    Header language, one of ['en', 'ger']
    :param keep_notes:                  Keeps note columns in data (Default: False)
    :param keep_index:                  Keeps original indices of eval and test data (Default: False)
    :param keep_label:                  Keeps the gold standard label in the X_data (Default: False)
    :param keep_original_numeric:       Duplicates numeric columns before Preprocessing, appends '_orig' to column names
    :param kwargs:                      Given to Preprocessors (first option is default):
                                            NumericalZscorePreprocessor
                                                nan_bucket: [False, True]
                                            NumericalMinMaxPreprocessor
                                                nan_bucket: [False, True]
                                            NumericalQuantizationPreprocessor
                                                encode: ['onehot', 'ordinal']
    """
    print("Loading data...")
    X_train, X_eval, X_test = load_erp_splits(source_folder=source_folder,
                                              dataset_name=dataset_name,
                                              language=language)

    # initializing preprocessors:
    if categorical_preprocessing == 'onehot':
        cat_preprocessor = CategoricalOneHotPreprocessor()
    else:
        cat_preprocessor = None

    if numeric_preprocessing == 'zscore':
        num_preprocessor = NumericalZscorePreprocessor(**kwargs)
    elif numeric_preprocessing == 'minmax':
        num_preprocessor = NumericalMinMaxPreprocessor(**kwargs)
    elif numeric_preprocessing == 'buckets':
        num_preprocessor = NumericalQuantizationPreprocessor(**kwargs)
    elif numeric_preprocessing == 'None':
        num_preprocessor = None
    else:
        raise ValueError(
            f"Variable preprocessing_format needs to be one of "
            f"['zscore', 'buckets', 'minmax'] but was: {numeric_preprocessing}")

    # Preprocessing of train-eval-test data
    cat_cols, num_cols = get_column_dtypes(source_folder, language=language)
    X_train = preprocessing(data=X_train,
                            cat_preprocessor=cat_preprocessor,
                            num_preprocessor=num_preprocessor,
                            cat_cols=cat_cols,
                            num_cols=num_cols,
                            fit_new=True,
                            keep_original_numeric=keep_original_numeric)
    X_eval = preprocessing(data=X_eval,
                           cat_preprocessor=cat_preprocessor,
                           num_preprocessor=num_preprocessor,
                           cat_cols=cat_cols,
                           num_cols=num_cols,
                           fit_new=False,
                           keep_original_numeric=keep_original_numeric)
    X_test = preprocessing(data=X_test,
                           cat_preprocessor=cat_preprocessor,
                           num_preprocessor=num_preprocessor,
                           cat_cols=cat_cols,
                           num_cols=num_cols,
                           fit_new=False,
                           keep_original_numeric=keep_original_numeric)

    note_cols = ["Belegnummer", "Position", "Transaktionsart", "Erfassungsuhrzeit"]
    if not keep_notes:
        X_train = X_train.drop(note_cols, axis=1)
        X_eval = X_eval.drop(note_cols, axis=1)
        X_test = X_test.drop(note_cols, axis=1)
    else:  # Reorder columns to have notes in front
        new_order = note_cols + [col for col in X_train.columns if col not in note_cols]
        X_train = X_train[new_order]
        X_eval = X_eval[new_order]
        X_test = X_test[new_order]

    if not keep_index:
        X_train = X_train.reset_index(drop=True)
        X_eval = X_eval.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

    # Drop Labels
    y_eval = X_eval["Label"]
    y_test = X_test["Label"]
    if not keep_label:
        X_train = X_train.drop(["Label"], axis=1)
        X_eval = X_eval.drop(["Label"], axis=1)
        X_test = X_test.drop(["Label"], axis=1)

    return X_train, X_eval, X_test, y_eval, y_test


def load_erp_dataset(ds_path, column_info, language):
    """
    Loads pandas dataframe from csv and changes language of headers based on header names in column_info
    :param ds_path:             String path to dataset .csv file
    :param column_info_path:    String path to column_info.csv file that includes header translations
    :param language:            String name of language row to use from column_info.csv file
    """
    dataset = pd.read_csv(ds_path, encoding='ISO-8859-1')
    if language not in column_info.columns.values:
        raise ValueError(f'Dataset language needs to be in column_info.csv index.\n'
                         f'Should be one of: {column_info.columns.values} but was: {language}')
    dataset.columns = column_info[language].values.T.reshape(column_info[language].shape[0])
    return dataset


def load_erp_splits(source_folder, dataset_name, language='en'):
    """
    Loads the erp system datasets joined together from BSEG and RSEG tables
    :param source_folder:   String path to data folder
    :param dataset_name:    String name of specific dataset folder
    :param language:        String name of language row to use from column_info.csv file
    """
    if 'ex1' in dataset_name:
        folder = 'ex1'
        benign_path = os.path.join(source_folder, folder, 'normal_2.csv')
        fraud1_path = os.path.join(source_folder, folder, 'fraud_2.csv')
        fraud2_path = os.path.join(source_folder, folder, 'fraud_3.csv')
        column_info = pd.read_csv(os.path.join(source_folder, folder, 'column_information.csv'),
                                  index_col=0,
                                  header=0).T
        benign = load_erp_dataset(ds_path=benign_path, column_info=column_info, language=language)
        fraud1 = load_erp_dataset(ds_path=fraud1_path, column_info=column_info, language=language)
        fraud2 = load_erp_dataset(ds_path=fraud2_path, column_info=column_info, language=language)

        X_train = benign
        X_eval = fraud1
        X_test = fraud2
    elif 'ex2' in dataset_name:
        folder = 'ex2'
        benign_path = os.path.join(source_folder, folder, 'normal_1.csv')
        fraud_path = os.path.join(source_folder, folder, 'fraud_1.csv')
        column_info = pd.read_csv(os.path.join(source_folder, folder, 'column_information.csv'),
                                  index_col=0,
                                  header=0).T
        benign = load_erp_dataset(ds_path=benign_path, column_info=column_info, language=language)
        fraud = load_erp_dataset(ds_path=fraud_path, column_info=column_info, language=language)

        X_train = benign
        # 50%-50%, making sure to not cut single accounting documents
        X_eval = fraud.iloc[19716:]  # last 50 % as eval data (4 frauds)
        X_test = fraud.iloc[:19716]  # first 50 % as test data (6 frauds)
    else:
        raise ValueError(f'Expected dataset_name string to contain one of '
                         f'["ex1", "ex2"] but was {dataset_name}')

    return X_train, X_eval, X_test


def preprocessing(data,
                  cat_preprocessor,
                  num_preprocessor,
                  cat_cols,
                  num_cols,
                  fit_new,
                  keep_original_numeric=False):
    """
    One-hot encoding and standardization for BSEG_RSEG.
    """
    # Categorical Preprocessing
    data[cat_cols] = data[cat_cols].astype(str)
    if cat_preprocessor:
        data_cat = data[cat_cols]
        data = data.drop(cat_cols, axis=1)
        if fit_new:
            cat_preprocessor.fit(data_cat)
        data_idx = data_cat.index
        data_cat = cat_preprocessor.transform(data_cat.reset_index(drop=True))
        data_cat.index = data_idx
        data = data.join(data_cat)

    # Numerical Preprocessing
    data_num = data[num_cols]
    if keep_original_numeric:
        data = data.rename({name: name + '_orig' for name in num_cols}, axis=1)
    else:
        data = data.drop(num_cols, axis=1)
    if num_preprocessor:
        if fit_new:
            num_preprocessor.fit(data_num)
        data_idx = data_num.index
        data_num = num_preprocessor.transform(data_num.reset_index(drop=True))
        data_num.index = data_idx
    data = data.join(data_num)

    return data
