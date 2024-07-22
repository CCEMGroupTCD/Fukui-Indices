import os
import filecmp
import warnings
import joblib
from typing import Union
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning

from src.utils.input_output import write_to_csv
from src.utils.utils import are_dir_trees_equal

warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')



class RunML(object):

    def __init__(self, df, models, features, target, CV_cols, scores, outdir, runtype, group, xscaler=None, yscaler=None):
        """
        Run a machine learning pipeline.
        :param df: DataFrame with the data. One row for each sample.
        :param models: Dictionary with the models to run. Keys are the model names, values are the models.
        :param features: List of feature names (columns in the df) to use for training the models.
        :param target: Name of the target column in the df.
        :param CV_cols: List of column names in the df that contain the train/test splits for cross validation.
        :param scores: Dictionary with the scores to calculate. Keys are the score names, values are the score functions.
        :param outdir: Path to the output directory.
        :param runtype: Type of the run, either 'regression', 'classification', or 'proba_classification'.
        :param group: Name of the column in the df that contains the group information. If None, no grouping is done.
        :param xscaler: Scaler for the features. If None, no scaling is done.
        :param yscaler: Scaler for the target. If None, no scaling is done. Can only be used for regression runs.
        """
        self.df = df
        self.models = models
        self.features = features
        self.target = target
        self.CV_cols = CV_cols
        self.init_x_scaler = xscaler
        self.init_y_scaler = yscaler
        self.scores = scores
        self.outdir = Path(str(outdir))
        self.runtype = runtype
        self.group_colname = group

        self.n_data = len(df)
        self.n_features = len(features)
        self.n_models = len(models)
        self.n_CV_splits = len(CV_cols)

        self.all_scores_outpath = Path('scores.csv')
        self.all_data_outpath = Path('data.csv')
        self.all_models_dir = Path('models')
        self.all_plots_dir = Path('plots')

        self.original_dir = None
        self.df_all_scores = None
        self.trained_models = None

        # Check if inputs make sense
        if self.runtype not in ['regression', 'classification', 'proba_classification']:
            raise ValueError(f'runtype must be either "regression", "classification", or "proba_classification", but is "{self.runtype}".')
        if self.runtype in ['classification', 'proba_classification'] and self.init_y_scaler is not None:
            raise ValueError('y_scaler must be None for classification runs.')

    def run(self):
        """
        Run all models on all CV splits and print scores.
        """
        # Create the output directory.
        if not self.outdir.exists():
            raise ValueError(f'Output directory {self.outdir} does not exist.')

        # Change to the output directory and save the original directory to be able to go back.
        self.original_dir = os.getcwd()
        os.chdir(self.outdir)
        print(f'Directory of run: "{self.outdir}"')

        # Print some infos about the input data.
        self.print_input_data_infos()

        print('Train models and calculate scores.')

        df_all_scores = []
        self.trained_models = {}
        for model_name, model in self.models.items():
            for CV_col in tqdm(self.CV_cols, desc=f'train all {model_name}'):
                # Very important to have a new model for each run! Otherwise, they might influence each other.
                model = deepcopy(model)
                model = self.train_model(model_name, model, CV_col)
                trained_model_name = f'{model_name}_{CV_col}'
                self.trained_models[trained_model_name] = model

            # Calculate metrics for the model.
            df_model_scores = self.calculate_scores(model_name)
            df_all_scores.append(df_model_scores)
        self.df_all_scores = pd.concat(df_all_scores, axis=0)

        self.save_data()

        # Make plots. Not yet implemented.
        self.make_plots()

        # Go back to the original directory.
        os.chdir(self.original_dir)

        return

    def print_input_data_infos(self):
        """
        Print some infos about the input data.
        :return:
        """
        n_train_samples = int((self.df[self.CV_cols] == 'train').sum(axis=0).mean())
        n_test_samples = int((self.df[self.CV_cols] == 'test').sum(axis=0).mean())
        print('Input data infos:')
        print(f'   Data size: {self.n_data}')
        print(f'   Train data size: {n_train_samples}')
        print(f'   Test data size: {n_test_samples}')
        print(f'   Num features: {self.n_features}')
        if self.n_features < 10:
            print(f'   Features: {self.features}')
        print(f'   Target: {self.target}')
        print(f'   Num CVs: {self.n_CV_splits}')

        return

    def save_data(self):
        """
        Save the data to disk.
        """
        print('Save data to disk.')
        comment = 'All scores of the run.'
        write_to_csv(df=self.df_all_scores, comment=comment, output_path=self.all_scores_outpath, verbose=False)
        print(f'\t- Saved scores to {self.all_scores_outpath.name}.')

        # Save the data.
        comment = 'Data used for training and testing the models. One row for each sample.'
        write_to_csv(df=self.df, comment=comment, output_path=self.all_data_outpath, verbose=False)
        print(f'\t- Saved data to {self.all_data_outpath.name}.')

        # Save the models.
        self.all_models_dir.mkdir()
        for model_name, model in self.trained_models.items():
            outpath = Path(self.all_models_dir, f'{model_name}.pkl')
            joblib.dump(model, outpath)
        print(f'\t- Saved models to {self.all_models_dir.name}.')

        return

    def get_y_true(self, CV_col, target=None) -> np.ndarray:
        """
        Get the true values for a given CV column.
        """
        if target is None:
            assert isinstance(self.target, str)
            target = self.target

        y_true = self.df.loc[self.df[CV_col] == 'test', target].to_numpy()

        return y_true

    def get_y_pred(self, CV_col, model_name, target=None) -> np.ndarray:
        """
        Get the predictions for a given model and CV column.
        """
        if target is None:
            assert isinstance(self.target, str)
            target = self.target

        pred_col = self.get_pred_colname(model_name, CV_col, target)
        y_pred = self.df.loc[self.df[CV_col] == 'test', pred_col].to_numpy()

        return y_pred

    def make_plots(self):
        """
        Make plots.
        :return: None
        """
        print('Make plots.')

        # Make the plots directory.
        self.all_plots_dir.mkdir()

        # parity plots
        for model_name in self.models.keys():
            for CV_col in self.CV_cols:
                y_true = self.get_y_true(CV_col)
                y_pred = self.get_y_pred(CV_col, model_name)
                outpath = Path(self.all_plots_dir, f'{model_name}_{CV_col}_parity.png')
                self.plot_parity_plot(y_true, y_pred, outpath)
        print(f'\t- Saved parity plots in {self.all_plots_dir.name}.')

        return

    def plot_parity_plot(self, y_true, y_pred, outpath):
        """
        Plot a parity plot.
        """
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.5)
        ax.plot([y_true.min(), y_pred.min()], [y_true.max(), y_pred.max()], color='k', linestyle='-')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Parity plot')
        fig.savefig(outpath)
        plt.close(fig)

        return

    def train_model(self, model_name, model, CV_col):
        """
        Run a single model on a single CV split.
        """
        X = self.df[self.features].to_numpy()

        # Get the training and test data.
        is_train = self.df[CV_col] == 'train'
        is_test = self.df[CV_col] == 'test'
        train_data = self.df[is_train]
        test_data = self.df[is_test]

        # Get the features and targets.
        X_train = train_data[self.features].to_numpy()
        y_train = train_data[self.target].to_numpy()
        X_test = test_data[self.features].to_numpy()

        # Scale the features and targets. Fit the scaler only on the training data to prevent data leakage.
        if self.init_x_scaler is not None:
            self.x_scaler = deepcopy(self.init_x_scaler).fit(X_train)
            X_train = self.x_scaler.transform(X_train)
            X_test = self.x_scaler.transform(X_test)
        if self.init_y_scaler is not None:
            self.y_scaler = deepcopy(self.init_y_scaler).fit(y_train.reshape(-1, 1))
            y_train = self.y_scaler.transform(y_train.reshape(-1, 1)).ravel()

        # Fit the model.
        model.fit(X_train, y_train)

        # Predict the test data.
        if self.runtype in ['regression', 'classification']:
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
        elif self.runtype == 'proba_classification':
            if set(y_train) != {0, 1}:
                y_train_values = set(y_train)
                raise ValueError(f'y_train must only contain 0 or 1 for runtype=`proba_classification`. Currently, y_train contains these unique values: {y_train_values}.')
            index_of_class_1 = model.classes_.tolist().index(1)
            y_pred_test = model.predict_proba(X_test)[: ,index_of_class_1]
            y_pred_train = model.predict_proba(X_train)[: ,index_of_class_1]


        # Unscale the features and targets.
        if self.init_x_scaler is not None:
            X_train = self.x_scaler.inverse_transform(X_train)
            X_test = self.x_scaler.inverse_transform(X_test)
        if self.init_y_scaler is not None:
            y_train = self.y_scaler.inverse_transform(y_train.reshape(-1, 1))
            y_pred_test = self.y_scaler.inverse_transform(y_pred_test.reshape(-1, 1))
            y_pred_train = self.y_scaler.inverse_transform(y_pred_train.reshape(-1, 1))

        # Save the predictions in the dataframe.
        pred_colname = self.get_pred_colname(model_name, CV_col, self.target)
        self.df.loc[is_test, pred_colname] = y_pred_test
        self.df.loc[is_train, pred_colname] = y_pred_train

        return model

    def get_pred_colname(self, model_name, CV_col, target) -> str:
        """
        Get the name of the column for the predictions.
        :param model_name:
        :param CV_col:
        :param target:
        :return:
        """
        return f'pred_{target}_{model_name}_{CV_col}'

    def calculate_scores(self, model_name) -> pd.DataFrame:
        """
        Calculate the scores for a single model.
        :param model_name: Name of the model to calculate scores for.
        :return: DataFrame of scores for the model.
        """
        all_scores = []
        for score_name in self.scores:
            score_func = self.scores[score_name]
            for CV in ['train', 'test']:
                for CV_col in self.CV_cols:
                    pred_colname = self.get_pred_colname(model_name, CV_col, self.target)

                    is_CV = self.df[CV_col] == CV
                    y_true = self.df.loc[is_CV, self.target]
                    y_pred = self.df.loc[is_CV, pred_colname]
                    group = self.df.loc[is_CV, self.group_colname]
                    try:
                        score_value = score_func(y_true=y_true, y_pred=y_pred, group=group)
                    except TypeError:
                        score_value = score_func(y_true=y_true, y_pred=y_pred)
                        warnings.warn(f'Function {score_func} does not take group as an argument. Ignoring group={self.group_colname} when calculating score {score_func}.', UserWarning)

                    all_scores.append({
                                        'value': score_value,
                                        'score': score_name,
                                        'CV': CV,
                                        'CV_col': CV_col,
                                        'model': model_name,
                                        'target': self.target
                                        })
        df_scores = pd.DataFrame(all_scores)

        groupby = ['score', 'CV', 'model', 'target']
        df_stats = df_scores.groupby(groupby)['value'].agg(['mean', 'sem']).reset_index()
        df_scores = df_scores.merge(df_stats, on=groupby)

        # Print the scores.
        print(f'{model_name} scores:'.ljust(12) + 'train'.center(20) + 'test'.center(20))
        for score_name in self.scores:
            # Calculate the mean and SEM for the train and test scores.
            train = df_scores[(df_scores['score'] == score_name) & (df_scores['CV'] == 'train')]
            train_mean = train['mean'].values[0]
            train_sem = train['sem'].values[0]
            test = df_scores[(df_scores['score'] == score_name) & (df_scores['CV'] == 'test')]
            test_mean = test['mean'].values[0]
            test_sem = test['sem'].values[0]

            # Print the scores in a nice format.
            score_str = f'\t{score_name}:'.ljust(10)
            train_str = f'{train_mean:.2f}±{train_sem:.2f}'.center(20)
            test_str = f'{test_mean:.2f}±{test_sem:.2f}'.center(20)
            print(score_str + train_str + test_str)

        return df_scores

    def check_if_output_same_as_reference(self, reference_run: Union[str, Path], detailed: bool = False):
        """
        Check if the output of this run is the same as the output of a reference run. This is very useful when refactoring/writing new code, to make sure that one always knows exactly what of the output changes.
        :param reference_run: Path to the directory of the reference run.
        :param detailed: If True, csv files are compared by using pd.assert_frame_equal, which will print a detailed error message if the files are not the same. If False, csv files are compared by using filecmp.cmp, which will only print a warning if the files are not the same.
        :return: None
        """
        print(f'\nChecking if output is still the same as reference run "{reference_run}"...')
        reference_run = Path(reference_run)
        if not reference_run.exists():
            print(f'Reference run "{reference_run}" does not exist. Exit without checking output files.')
            return
        elif reference_run.absolute() == self.outdir.absolute():
            print(f'Reference run "{reference_run}" is the same as the current run. Exit without checking output files.')
            return

        all_files_in_run = sorted([Path(f).name for f in os.listdir(self.outdir) if (Path(self.outdir, f).is_file() and not Path(f).name.startswith('.'))])
        all_dirs_in_run = sorted([Path(f).name for f in os.listdir(self.outdir) if (Path(self.outdir, f).is_dir() and not Path(f).name.startswith('.'))])

        for f in all_files_in_run:
            if not Path(reference_run, f).exists():
                print(f'\t- WARNING: Output file {f} does not exist in reference run.')
            else:

                if detailed and f.endswith('.csv'):
                    print(f'\t- Checking {f}')
                    df_new = pd.read_csv(Path(self.outdir, f))
                    df_old = pd.read_csv(Path(reference_run, f))
                    pd.testing.assert_frame_equal(df_new, df_old, check_like=True)
                else:
                    same = filecmp.cmp(Path(self.outdir, f), Path(reference_run, f), shallow=False)
                    if not same:
                        print(f'\t- WARNING: Output is not the same for {f}')
                    else:
                        print(f'\t- All good: {f}')

        for d in all_dirs_in_run:
            same = are_dir_trees_equal(Path(self.outdir, d), Path(reference_run, d))
            if not same:
                print(f'\t- WARNING: Output is not the same for {d}')
            else:
                print(f'\t- All good: {d}')

        return