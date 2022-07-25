
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from anomaly_detection.autoencoder import Autoencoder
from dask_ml.wrappers import ParallelPostFit

from anomaly_detection.nalu import NaluAE
from data.util import load_and_preprocess
from xai.xai_shap import anomaly_shap_values
from xai.util import xai_to_categorical


def load_best_detector(model, dataset_name):
    if model == 'OCSVM':
        print("Loading SVM...")
        detector = joblib.load(f'./outputs/models/OC_SVM_{dataset_name}.pkl')
        detector = ParallelPostFit(estimator=DaskOCSVM(detector))
    elif model == 'AE':
        best_params = {'cpus': 8,
                       'n_inputs': 240 if dataset_name == 'ex2' else 243,
                       'n_layers': 2,
                       'n_bottleneck': 32,
                       'epochs': 100,
                       'batch_size': 32,
                       'learning_rate': 1e-5,
                       'shuffle': True,
                       'verbose': 2}
        detector = Autoencoder(**best_params)
        detector = detector.load(f'outputs/models/AE_{dataset_name}')
    elif model == 'NALU':
        best_params = {'cpus': 8,
                       'n_inputs': 240 if dataset_name == 'ex2' else 243,
                       'n_layers': 1,
                       'n_bottleneck': 32,
                       'epochs': 100,
                       'batch_size': 32,
                       'learning_rate': 1e-4,
                       'shuffle': True,
                       'verbose': 2}
        detector = NaluAE(**best_params)
        detector = detector.load(f'outputs/models/NALU_{dataset_name}')
    elif model == 'IF':
        from sklearn.ensemble import IsolationForest
        best_params = {'n_estimators': 128,
                       'max_samples': 1.,
                       'max_features': 0.4,
                       'bootstrap': False,
                       'n_jobs': -1,
                       'random_state': 0}
        detector = IsolationForest(**best_params)
    elif model == 'PCA':
        from anomaly_detection.pyod_wrapper import PyodDetector
        best_params = {'n_components': 0.95,
                       'whiten': True,
                       'random_state': 0,
                       'weighted': False,
                       'standardization': False,
                       'algorithm': 'PCA'}
        detector = PyodDetector(**best_params)
    else:
        raise ValueError(f"Expected 'model' to be one of ['OCSVM', 'AE', 'NALU', 'PCA', 'IF'], but was {model}")
    return detector


class DaskOCSVM:
    """Small wrapper to trick dask_ml into parallelizing anomaly detection methods"""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.score_samples(X)


def get_expl_scores(explanation, gold_standard, dataset_name, score_type='auc_roc'):
    """Calculate AUC-ROC score for each sample individually, report mean and std"""
    scores = []
    for i, row in explanation.iterrows():
        # Explanation values for each feature treated as likelihood of anomalous feature
        #  -aggregated to feature-scores over all feature assignments
        #  -flattened to match shape of y_true
        #  -inverted, so higher score means more anomalous
        y_score = xai_to_categorical(expl_df=pd.DataFrame(explanation.loc[i]).T,
                                     dataset_name=dataset_name,
                                     language='ger').values.flatten() * -1
        # Calculate score
        if score_type == 'auc_roc':
            scores.append(roc_auc_score(y_true=gold_standard.loc[i], y_score=y_score))
        elif score_type == 'auc_pr':
            scores.append(average_precision_score(y_true=gold_standard.loc[i], y_score=y_score))
    return np.mean(scores), np.std(scores)


def evaluate_expls(dataset, dataset_name, gold_standard_path, expl_folder, model, out_path, eval_score):
    """Calculate AUC-ROC score of highlighted important features"""
    expl = pd.read_csv(os.path.join(expl_folder, '{}_shap_{}.csv'.format(model, dataset)),
                       header=0, index_col=0)
    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    # Load gold standard explanations and convert to pd.Series containing
    # anomaly index & list of suspicious col names as values
    gold_expl = pd.read_csv(gold_standard_path, header=0, index_col=0, encoding='UTF8')
    gold_expl = (gold_expl == 'X').iloc[:, :-5]  # .apply(lambda x: list(x[x.values].index.values), axis=1)

    # Remove anomalies, only check the frauds
    if dataset_name == 'ex1':
        if 'fraud_2' in dataset:
            to_check = [19919, 19920, 19961, 19962, 27123, 27124, 13246, 13247, 13248, 13249, 13250, 13251,
                        35015, 35016, 35017, 35018, 660, 661, 662, 32227, 32228, 32229, 17419, 17420, 17421,
                        22709, 22710, 22711, 1207, 4252, 4253, 4254, 4255, 4256, 4257, 1196, 1197, 1198, 1199,
                        1200, 1201, 1202, 1203, 1204, 1205, 1206, 33138, 33139, 33140, 33141]
        elif 'fraud_3' in dataset:
            to_check = [2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 13759, 13760, 13765, 13766,
                        22205, 22206, 22211, 22212, 8755, 8756, 8757, 8758, 8759, 8760, 8761, 8762, 8763, 8764,
                        8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772, 8773, 8774, 8775, 8776, 35046, 35047,
                        35048, 35049, 684, 685, 5064, 5065, 15745, 15746, 15747, 15748, 15749, 15750, 15751,
                        15752, 15753, 15754, 15755, 15756, 22424, 22425, 22426, 22427, 22428, 22429, 22430,
                        22431, 22432, 22433, 22434, 22435, 361, 362, 1035, 1036, 1037, 1038, 36376, 36377,
                        36378, 36379, 36802, 36803, 36804, 36805]
        else:
            raise ValueError("Expected either 'run2' or 'run3' to be specified in variable dataset_name")
    elif dataset_name == 'ex2':
        to_check = [17930, 17931, 19067, 19068, 19080, 19081, 23096, 23097, 23098, 23099, 24806, 24807,
                    24810, 24811, 24812, 24813, 24814, 24815, 24816, 24817, 24818, 24819, 24822, 24823]
    else:
        raise ValueError(f"Variable dataset_name needs to be one of ['ex1', 'ex2'] but was: {dataset_name}")

    score_mean, score_std = get_expl_scores(explanation=expl.loc[to_check],
                                            gold_standard=gold_expl.loc[to_check],
                                            dataset_name=dataset_name,
                                            score_type=eval_score)
    out_dict = {'model': model,
                'dataset': dataset,
                f'{eval_score}-mean': score_mean,
                f'{eval_score}-std:': score_std,
                'found': len(to_check)}
    print(out_dict)
    # save outputs to combined result csv file
    if out_path:
        if os.path.exists(out_path):
            out_df = pd.read_csv(out_path, header=0)
        else:
            out_df = pd.DataFrame()
        out_df = out_df.append(out_dict, ignore_index=True)
        out_df.to_csv(out_path, index=False)
    return out_dict


def explain_anomalies(expl_folder,
                      model='AE',
                      numeric_preprocessing='bucket',
                      out_path=None,
                      eval_score='auc_pr',
                      **kwargs):
    """
    :param expl_folder:     Str path to folder to write/read explanations to/from
    :param model:           Str type of model to load, one of ['AE', 'OCSVM', 'NALU', 'IF', 'PCA']
    :param numeric_preprocessing:   Str type of numeric preprocessing, one of ['buckets', 'minmax', 'zscore', 'None']
    :param kwargs:          Additional keyword args directly for numeric preprocessors during data loading
    """
    dataset_name = 'ex1' if 'ex1' in dataset else 'ex2'
    X_train, X_eval, X_test, y_eval, y_test = load_and_preprocess(source_folder='./data',
                                                                  dataset_name=dataset_name,
                                                                  numeric_preprocessing=numeric_preprocessing,
                                                                  keep_index=True,
                                                                  **kwargs)
    # find gold standard explanations for anomalous cases
    if dataset_name == 'ex1' and 'fraud_2' in dataset:
        ds_file = 'fraud_2_expls.csv'
    elif dataset_name == 'ex1' and 'fraud_3' in dataset:
        ds_file = 'fraud_3_expls.csv'
    elif dataset_name == 'ex2':
        ds_file = 'fraud_1_expls.csv'
    else:
        raise ValueError('Variable dataset needs to note which gold standard explanation to use (experiment + dataset)')
    gold_expl_path = f'data/{dataset_name}/{ds_file}'

    # reads index from anomalous samples from gold standard
    if dataset_name == 'ex1' and 'fraud_2' in dataset:
        frauds = [19919, 19920, 19961, 19962, 27123, 27124, 13246, 13247, 13248, 13249, 13250, 13251,
                  35015, 35016, 35017, 35018, 660, 661, 662, 32227, 32228, 32229, 17419, 17420, 17421,
                  22709, 22710, 22711, 1207, 4252, 4253, 4254, 4255, 4256, 4257, 1196, 1197, 1198, 1199,
                  1200, 1201, 1202, 1203, 1204, 1205, 1206, 33138, 33139, 33140, 33141]
        X_expl = X_eval.loc[frauds]
    elif dataset_name == 'ex1' and 'fraud_3' in dataset:
        frauds = [2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 13759, 13760, 13765, 13766,
                  22205, 22206, 22211, 22212, 8755, 8756, 8757, 8758, 8759, 8760, 8761, 8762, 8763, 8764,
                  8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772, 8773, 8774, 8775, 8776, 35046, 35047,
                  35048, 35049, 684, 685, 5064, 5065, 15745, 15746, 15747, 15748, 15749, 15750, 15751,
                  15752, 15753, 15754, 15755, 15756, 22424, 22425, 22426, 22427, 22428, 22429, 22430,
                  22431, 22432, 22433, 22434, 22435, 361, 362, 1035, 1036, 1037, 1038, 36376, 36377,
                  36378, 36379, 36802, 36803, 36804, 36805]
        X_expl = X_test.loc[frauds]
    elif dataset_name == 'ex2':
        frauds = [17930, 17931, 19067, 19068, 19080, 19081, 23096, 23097, 23098, 23099, 24806, 24807,
                  24810, 24811, 24812, 24813, 24814, 24815, 24816, 24817, 24818, 24819, 24822, 24823]
        X_expl = pd.concat([X_eval, X_test]).loc[frauds]
    else:
        raise ValueError('Variable dataset needs to note which gold standard explanation to use (session + run)')

    print('Loading detector...')
    detector = load_best_detector(model=model, dataset_name=dataset_name)

    if not os.path.exists(os.path.join(expl_folder, '{}_{}_{}.csv'.format(model, 'shap', dataset))):
        print("Generating explanations...")
        out_template = os.path.join(expl_folder, '{}_{{}}_{}.csv'.format(model, dataset))
        anomaly_shap_values(X_anomalous=X_expl,
                            X_benign=X_train,
                            detector=detector,
                            out_template=out_template)

    print('Evaluating explanations...')
    if shard_data is None:
        out_dict = evaluate_expls(dataset=dataset,
                                  dataset_name=dataset_name,
                                  gold_standard_path=gold_expl_path,
                                  expl_folder=expl_folder,
                                  model=model,
                                  out_path=out_path,
                                  eval_score=eval_score)
        return out_dict


if __name__ == '__main__':
    """
    Argparser needs to accept all possible param_search arguments, but only passes given args to params.
    """

    parser = ArgumentParser()
    parser.add_argument(f'--shard_data', type=int, default=None)
    args_dict = vars(parser.parse_args())

    if 'shard_data' in args_dict:
        shard_data = args_dict.pop('shard_data')
    else:
        shard_data = None

    # ['AE', 'OCSVM', 'NALU', 'IF', 'PCA']
    models = ['NALU']
    # ['ex1_fraud_2', 'ex1_fraud_3', 'ex2_fraud_1']
    datasets = ['ex1_fraud_2']
    add_to_summary = False
    eval_score = 'auc_roc'

    for model in models:
        for dataset in datasets:
            explain_anomalies(expl_folder=f'./outputs/explanation/',
                              dataset=dataset,
                              model=model,
                              numeric_preprocessing='buckets',
                              shard_data=shard_data,
                              out_path='./outputs/explanation/summary.csv' if add_to_summary else None,
                              eval_score=eval_score)
