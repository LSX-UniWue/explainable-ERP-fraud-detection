
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from data.util import load_and_preprocess
from anomaly_detection.util import evaluate_detector
from anomaly_detection.nalu import NaluAE
from anomaly_detection.autoencoder import Autoencoder
from anomaly_detection.pyod_wrapper import PyodDetector


def detect_anomalies(algorithm, dataset_name, experiment_name, numeric='zscore', numeric_nan_bucket=False,
                     params=None, output_scores=False, seed=0):
    print("Loading data...")
    X_train, X_eval, X_test, y_eval, y_test = load_and_preprocess(source_folder='./data',
                                                                  dataset_name=dataset_name,
                                                                  numeric_preprocessing=numeric,
                                                                  nan_bucket=numeric_nan_bucket)

    if algorithm == 'IsolationForest':
        detector_class = IsolationForest
        params['random_state'] = seed
    elif algorithm == 'OneClassSVM':
        detector_class = OneClassSVM
    elif algorithm == 'Autoencoder':
        detector_class = Autoencoder
        params['n_inputs'] = X_train.shape[1]
        params['seed'] = seed
    elif algorithm == 'NALU':
        detector_class = NaluAE
        params['n_inputs'] = X_train.shape[1]
        params['seed'] = seed
    elif algorithm in ['PCA']:
        detector_class = PyodDetector
        params['algorithm'] = algorithm
    else:
        raise ValueError(f"Variable algorithm was: {algorithm}")

    # path to save autoencoder model
    save_path = None
    if 'save_path' in params.keys():
        save_path = params.pop('save_path')

    # Training
    detector = detector_class(**params).fit(X_train)

    # Anomaly classification outputs
    scores_eval, eval_dict = evaluate_detector(detector=detector, X=X_eval, y=y_eval)
    scores_test, test_dict = evaluate_detector(detector=detector, X=X_test, y=y_test)
    eval_dict = {key + '_eval': val for key, val in eval_dict.items()}
    test_dict = {key + '_test': val for key, val in test_dict.items()}
    out_df = pd.DataFrame()
    out_df = out_df.append({**params, **eval_dict, **test_dict}, ignore_index=True)
    out_df.to_csv(os.path.join('./outputs/', experiment_name + '.csv'), index=False)
    print(out_df)

    if output_scores:
        score_df = pd.concat([pd.Series(ndarr, name=name) for name, ndarr in
                              {'scores_eval': scores_eval.values, 'y_eval': y_eval,
                               'scores_test': scores_test.values, 'y_test': y_test}.items()], axis=1)
        score_df.to_csv(os.path.join('./outputs/', experiment_name + '_scores.csv'), index=False)

    if save_path:
        if algorithm in ['Autoencoder', 'NALU']:
            detector.save(save_path=save_path)
        elif algorithm in ['OneClassSVM']:
            import joblib
            joblib.dump(detector, f'{save_path}.pkl')


if __name__ == '__main__':
    """
    Argparser needs to accept all possible param_search arguments, but only passes given args to params.
    """
    str_args = ('env', 'experiment_name', 'algorithm', 'dataset_name', 'kernel', 'save_path', 'numeric')
    float_args = ('tol', 'nu', 'max_samples', 'max_features', 'coef0', 'gamma', 'alpha', 'learning_rate')
    int_args = ('n_estimators', 'n_jobs', 'random_state', 'degree', 'cpus',
                'n_neighbors', 'leaf_size',
                'n_layers', 'n_bottleneck', 'epochs', 'batch_size', 'verbose', 'seed', 'cache_size', 'max_iter')
    bool_args = ('bootstrap', 'novelty', 'shuffle', 'output_scores', 'shrinking', 'k', 'numeric_nan_bucket')
    parser = ArgumentParser()
    for arg in str_args:
        parser.add_argument(f'--{arg}')
    for arg in int_args:
        parser.add_argument(f'--{arg}', type=int)
    for arg in float_args:
        parser.add_argument(f'--{arg}', type=float)
    for arg in bool_args:
        parser.add_argument(f'--{arg}', action='store_true')
    args_dict = vars(parser.parse_args())

    env = args_dict.pop('env')
    algorithm = args_dict.pop('algorithm')
    dataset_name = args_dict.pop('dataset_name')
    experiment_name = args_dict.pop('experiment_name')
    output_scores = args_dict.pop('output_scores')
    numeric = args_dict.pop('numeric')
    numeric_nan_bucket = args_dict.pop('numeric_nan_bucket')
    seed = args_dict.pop('seed')
    np.random.seed(seed)

    params = {key: val for key, val in args_dict.items() if val}  # remove entries with None values
    detect_anomalies(algorithm=algorithm,
                     dataset_name=dataset_name,
                     experiment_name=experiment_name,
                     params=params,
                     numeric=numeric,
                     numeric_nan_bucket=numeric_nan_bucket,
                     output_scores=output_scores,
                     seed=seed)
