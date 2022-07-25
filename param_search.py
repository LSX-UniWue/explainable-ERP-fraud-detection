import pandas as pd
from sklearn.model_selection import ParameterGrid

from run_detector import detect_anomalies


def get_param_grid(algorithm, seed, cpus):
    if algorithm == 'IsolationForest':
        return ParameterGrid({'n_estimators': [2 ** n for n in range(4, 9)],
                              'max_samples': [0.4, 0.6, 0.8, 1.],
                              'max_features': [0.4, 0.6, 0.8, 1.],
                              'bootstrap': [False],
                              'n_jobs': [-1]})

    elif algorithm == 'OneClassSVM':
        return ParameterGrid({"kernel": ['rbf'],
                              'gamma': [1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
                              'tol': [1e-3],
                              'nu': [0.2, 0.4, 0.6, 0.8],
                              'shrinking': [True],
                              'cache_size': [500],
                              'max_iter': [-1],
                              'save_path': [None]})

    elif algorithm == 'Autoencoder':
        return ParameterGrid({'cpus': [cpus],
                              'n_layers': [4, 3, 2],
                              'n_bottleneck': [16, 32],
                              'epochs': [100],
                              'batch_size': [32],
                              'learning_rate': [1e-5, 1e-4, 1e-3],
                              'shuffle': [True],
                              'verbose': [2],
                              'save_path': [None]})

    elif algorithm == 'NALU':
        return ParameterGrid({'cpus': [cpus],
                              'n_layers': [1, 2],
                              'n_bottleneck': [16, 32, 64, 128],
                              'epochs': [100],
                              'batch_size': [32],
                              'learning_rate': [1e-4, 1e-3, 1e-2],
                              'shuffle': [True],
                              'verbose': [2],
                              'save_path': [None]})

    elif algorithm == 'PCA':
        return ParameterGrid({'n_components': [0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
                              'whiten': [True, False],
                              'random_state': [seed],
                              'weighted': [True, False],
                              'standardization': [False]})

    else:
        raise ValueError(f"Variable algorithm was: {algorithm}")


if __name__ == '__main__':

    seeds = [0]  # [0], list(range(5))
    numeric = 'buckets'  # One of ['zscore', 'buckets', 'minmax']
    numeric_nan_bucket = False
    # One of ['Autoencoder', 'OneClassSVM', 'IsolationForest', 'PCA', 'NALU']
    algorithm = 'NALU'
    dataset_name = 'ex1'  # One of ['ex1', 'ex2']
    kernel = 'rbf'  # used for OneClassSVM
    out_template = f'{algorithm}_{dataset_name}_{{}}_local'

    for i, seed in enumerate(seeds):
        param_grid = get_param_grid(algorithm=algorithm, seed=seed, cpus=0)
        for j, params in enumerate(param_grid):
            detect_anomalies(algorithm=algorithm,
                             dataset_name=dataset_name,
                             experiment_name=out_template.format(str(i) + '_' + str(j)),
                             numeric=numeric,
                             numeric_nan_bucket=numeric_nan_bucket,
                             params=params,
                             output_scores=True,
                             seed=seed)
