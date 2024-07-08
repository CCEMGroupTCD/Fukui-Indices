"""
Utility functions for machine learning.
"""
from typing import Tuple, List, Union
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, ShuffleSplit, LeaveOneGroupOut, LeaveOneOut, GroupKFold, GroupShuffleSplit
import yaml
from pathlib import Path

def get_hparams(hparams_file: Union[str,Path]) -> dict:
    """
    Get hyperparameters from .yaml file.
    """
    hparams_file = Path(str(hparams_file))      # convert str to Path()
    if hparams_file.exists():
        hparams = yaml.load(open(hparams_file,"r"), Loader=yaml.FullLoader)
    else:
        raise ValueError(f'No hyperparameter file "{hparams_file}" found.')

    return(hparams)


def net_pattern(n_layers, base_size, end_size):
    """Calculates layer sizes for each layer from first and last layer so that the layer size continously increases/decreases.
    """
    if n_layers != 1:
        factor = (end_size / base_size) ** (1 / (n_layers - 1))
    else:
        factor = 1

    layer_sizes = [int(round(base_size * factor ** n)) for n in range(n_layers)]
    return (layer_sizes)


def load_data(
                dataset: Union[str,Path],
                features: List[str],
                targets: Union[str,List[str]],
                use_data_frac: Union[float,None] = None,
                shuffle: bool = True,
                reset_index: bool = True,
                header: int = 0,
                ) -> pd.DataFrame:
    """
    Load ML input data from csv file into pandas DataFrame.
    @param dataset: Path to csv file containing the data.
    @param features: List of features for double checking if all features are present.
    @param targets: List of targets for double checking if all targets are present.
    @param use_data_frac: Fraction of data to use, between 0 and 1. If None, all data is used.
    """
    if isinstance(targets, str):
        targets = [targets]

    data = pd.read_csv(dataset, header=header)

    # Reduce data to a fraction of the full data. Mainly useful for faster runtime for testing purposes.
    if use_data_frac is not None:
        # Check if use_data_frac is between 0 and 1:
        if use_data_frac <= 0 or use_data_frac >= 1:
            raise ValueError(f'use_data_frac must be between 0 and 1 or None, but is {use_data_frac}.')

        data = data.sample(frac=use_data_frac)

    # Shuffle data
    if shuffle:
        data = data.sample(frac=1)
        if reset_index:
            data = data.reset_index(drop=True)

    # Check if dataset contains all features and targets
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f'Feature "{feature}" not found in dataset. Possible features are: {data.columns}')
    for target in targets:
        if target not in data.columns:
            raise ValueError(f'Target "{target}" not found in dataset. Possible targets are: {data.columns}')

    return data

def name_CV_col(n):
    """
    Retuns the name pattern of the CV column of `data` that contains the train and test indices for the nth cross validation run.
    """
    return f'CV_{n}'

def get_train_test_splits(df, CV, n_reps, trainfrac=None, group=None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Gets train and test data doing the specified cross validation.
    """
    data_array = df.to_numpy()
    if CV == 'KFold':
        # Convert train fraction and number of total repetitions to KFold input parameters.
        if group is None:
            split = RepeatedKFold(n_splits=n_reps, n_repeats=1).split(data_array)
        else:
            groups = df[group].to_numpy()
            split = GroupKFold(n_splits=n_reps).split(data_array, groups=groups)

    elif CV == 'Random':
        assert trainfrac not in (0, 1), "ShuffleSplit won't understand that this is a fraction."
        assert trainfrac != None
        if group is None:
            split = ShuffleSplit(train_size=trainfrac, n_splits=n_reps).split(data_array)
        else:
            groups = df[group].to_numpy()
            split = GroupShuffleSplit(train_size=trainfrac, n_splits=n_reps).split(data_array, groups=groups)

    elif CV == 'LeaveOneOut':
        if group is None:
            split = LeaveOneOut().split(data_array)
        else:
            groups = df[group].to_numpy()
            split = LeaveOneGroupOut().split(data_array, groups=groups)
    elif CV.startswith('TestMolecule='):
        molname = CV.split('=')[-1]
        assert group in df.columns, f'Column name "{group}" not found in df.columns.'
        test_indices = df[df[group] == molname].index
        train_indices = df[df[group] != molname].index
        split = [(train_indices, test_indices)]
    elif CV == 'AsFile':
        # This means that the input csv already contains a column with the CV split.
        CV_cols = [col for col in df.columns if col.startswith('CV_')]
        return (df, CV_cols)
    else:
        raise ValueError(f'Unknown cross-validation method "{CV}".')


    # Create a column in df for each CV split and fill it with either 'train' or 'test'.
    CV_cols = []
    for i, (train_indices, test_indices) in enumerate(split):
        n_samples = len(df)
        assert n_samples == len(train_indices) + len(test_indices)

        empty = ''
        test_or_train = pd.Series(np.full(n_samples, empty))
        test_or_train[train_indices] = 'train'
        test_or_train[test_indices] = 'test'
        # So that I can be sure that in case the indices of df and series don't align it still just adds everything in the right order.
        test_or_train = list(test_or_train)

        colname = name_CV_col(i)
        df[colname] = test_or_train
        CV_cols.append(colname)

        assert all([df[colname].iloc[idx] == 'train' for idx in train_indices])
        assert all([df[colname].iloc[idx] == 'test' for idx in test_indices])
        assert not (df[colname] == empty).any(), 'Some of the columns are neither test nor train.'

    return (df, CV_cols)