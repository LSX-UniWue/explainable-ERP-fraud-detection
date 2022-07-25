
import os
import numpy as np
import pandas as pd

from anomaly_detection.util import get_mean_rank, get_max_rank, get_auc_pr_score, get_auc_roc_score


def parse_paramsearch_results(folder, approach, out_template):
    """parse_paramsearch_results(folder='./outputs', approach='OneClassSVM', out_path='./outputs/summary')"""
    out_dfs = []
    for filename in os.listdir(folder):
        if filename.startswith(approach) and filename.endswith('.csv') and not filename.endswith('_scores.csv'):
            out_dfs.append(pd.read_csv(os.path.join(folder, filename)))
    out_df = pd.concat(out_dfs).reset_index(drop=True)
    if 'tol' in out_df.columns:
        out_df = out_df[out_df['tol'] != 1e-4]
    out_df.to_csv(os.path.join(out_template.format(approach)), index=False)


def add_score(score_name, folder, approach):
    for f_name in os.listdir(folder):
        if f_name.startswith(approach) and f_name.endswith('_scores.csv'):  # find fitting score files
            scoring_df = pd.read_csv(os.path.join(folder, f_name))
            scores_eval = scoring_df['scores_eval'].dropna()
            scores_test = scoring_df['scores_test'].dropna()
            y_eval = scoring_df['y_eval'].dropna()
            y_test = scoring_df['y_test'].dropna()

            # add different scores
            if score_name == 'mean_rank':
                score_dict_eval = get_mean_rank(scores=scores_eval, y=y_eval)
                score_dict_test = get_mean_rank(scores=scores_test, y=y_test)
            elif score_name == 'max_rank':
                score_dict_eval = get_max_rank(scores=scores_eval, y=y_eval)
                score_dict_test = get_max_rank(scores=scores_test, y=y_test)
            elif score_name == 'auc_pr':
                score_dict_eval = get_auc_pr_score(scores=scores_eval, y=y_eval)
                score_dict_test = get_auc_pr_score(scores=scores_test, y=y_test)
            elif score_name == 'auc_roc':
                score_dict_eval = get_auc_roc_score(scores=scores_eval, y=y_eval)
                score_dict_test = get_auc_roc_score(scores=scores_test, y=y_test)
            else:
                raise ValueError(score_name)
            score_dict_eval = {key + '_eval': val for key, val in score_dict_eval.items()}
            score_dict_test = {key + '_test': val for key, val in score_dict_test.items()}
            score_dict = {**score_dict_eval, **score_dict_test}

            overview_path = os.path.join(folder, f_name[:-11] + '.csv')  # load respective overview .csv
            out_df = pd.read_csv(overview_path)
            out_df = out_df.drop(score_dict.keys(), axis=1, errors='ignore')  # drop previous scores if they exist
            out_df = pd.merge(out_df, pd.DataFrame(pd.Series(score_dict)).T, left_index=True, right_index=True)
            out_df.to_csv(overview_path, index=False)


def add_joint_score(score_name, folder, approach):
    """joins the eval and test sets and calculates joint scores"""
    for f_name in os.listdir(folder):
        if f_name.startswith(approach) and f_name.endswith('_scores.csv'):  # find fitting score files
            scoring_df = pd.read_csv(os.path.join(folder, f_name))
            scoring_eval = scoring_df[['scores_eval', 'y_eval']].dropna().rename(columns={'scores_eval': 'scores', 'y_eval': 'y'})
            scoring_test = scoring_df[['scores_test', 'y_test']].dropna().rename(columns={'scores_test': 'scores', 'y_test': 'y'})
            scorings = pd.concat([scoring_eval, scoring_test], ignore_index=True)
            scores = scorings['scores']
            y = scorings['y']

            # add different scores
            if score_name == 'mean_rank':
                score_dict = get_mean_rank(scores=scores, y=y)
            elif score_name == 'max_rank':
                score_dict = get_max_rank(scores=scores, y=y)
            elif score_name == 'auc_pr':
                score_dict = get_auc_pr_score(scores=scores, y=y)
            elif score_name == 'auc_roc':
                score_dict = get_auc_roc_score(scores=scores, y=y)
            else:
                raise ValueError(score_name)
            score_dict = {key + '_joint': val for key, val in score_dict.items()}

            overview_path = os.path.join(folder, f_name[:-11] + '.csv')  # load respective overview .csv
            out_df = pd.read_csv(overview_path)
            out_df = out_df.drop(score_dict.keys(), axis=1, errors='ignore')  # drop previous scores if they exist
            out_df = pd.merge(out_df, pd.DataFrame(pd.Series(score_dict)).T, left_index=True, right_index=True)
            out_df.to_csv(overview_path, index=False)


if __name__ == '__main__':

    task_name = 'ex1'
    approach = 'OneClassSVM'
    folder = './ex1'  # needs to end without /

    if task_name == 'session1':
        add_joint_score(score_name='max_rank', folder=folder, approach=approach)
        add_joint_score(score_name='mean_rank', folder=folder, approach=approach)
        add_joint_score(score_name='auc_pr', folder=folder, approach=approach)
        add_joint_score(score_name='auc_roc', folder=folder, approach=approach)
        parse_paramsearch_results(folder=folder, approach=approach,
                                  out_template=f'./summary/{task_name}/{approach}_{folder.split("/")[-1]}.csv')
    elif task_name == 'session2':
        add_score(score_name='max_rank', folder=folder, approach=approach)
        add_score(score_name='mean_rank', folder=folder, approach=approach)
        add_score(score_name='auc_pr', folder=folder, approach=approach)
        add_score(score_name='auc_roc', folder=folder, approach=approach)
        parse_paramsearch_results(folder=folder, approach=approach,
                                  out_template=f'./summary/{task_name}/{approach}_{folder.split("/")[-1]}.csv')
