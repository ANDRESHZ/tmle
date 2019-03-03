import os
import pickle
import numpy as np
import hyperopt
import sklearn

from typing import Optional
from functools import partial
from sklearn.model_selection import StratifiedKFold
from hyperopt import fmin, tpe, Trials, STATUS_OK


class ClassifierOptimizer(object):

    def __init__(
            self,
            classifier: sklearn.base.ClassifierMixin,
            space: dict,
            metric: sklearn.metrics
    ) -> None:
        """

        :param classifier:
        :param space:
        :param metric:
        """
        self.classifier = classifier
        self.space = space
        self.metric = metric

    def find_best_params(
            self,
            X: np.ndarray,
            y: np.ndarray,
            experiments_path: str,
            experiments_name: str,
            max_evals: int = 10,
            n_splits: int = 3,
            verbose: bool = True
    ) -> Optional[dict]:
        """

        :param X:
        :param y:
        :param experiments_path:
        :param experiments_name:
        :param max_evals:
        :param n_splits:
        :param verbose:
        :return:
        """
        if os.path.exists(os.path.join(experiments_path, '.'.join([experiments_name, 'hpopt']))):
            trials = pickle.load(open(os.path.join(experiments_path, '.'.join([experiments_name, 'hpopt'])), 'rb'))
            max_evals = len(trials.trials) + max_evals
        else:
            trials = Trials()

        try:
            best_params = fmin(
                fn=partial(
                    self.evaluate_params,
                    X=X, y=y, n_splits=n_splits, verbose=verbose
                ),
                space=self.space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials
            )
        except KeyboardInterrupt:
            # saving experiments on exit
            self._save_experiments_results(
                trials=trials,
                experiments_path=experiments_path,
                experiments_name=experiments_name
            )
            return None

        # save trials object for further usage
        self._save_experiments_results(
            trials=trials,
            experiments_path=experiments_path,
            experiments_name=experiments_name
        )

        return best_params

    def evaluate_params(
            self,
            clf_params: dict,
            X: np.ndarray,
            y: np.ndarray,
            n_splits: int = 3,
            verbose: bool = True
    ) -> dict:
        """

        :param X:
        :param y:
        :param clf_params:
        :return:
        """
        self.classifier.set_params(**clf_params)
        score_train, score_valid = [], []
        for train_idx, valid_idx in StratifiedKFold(n_splits=n_splits).split(X, y):
            x_train_fold, x_valid_fold = X[train_idx], X[valid_idx]
            y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]
            self.classifier.fit(x_train_fold, y_train_fold)
            score_train.append(self.metric(y_train_fold, self.classifier.predict(x_train_fold)))
            score_valid.append(self.metric(y_valid_fold, self.classifier.predict(x_valid_fold)))
        mean_score_train = np.mean(score_train)
        mean_score_valid = np.mean(score_valid)
        if verbose:
            msg = 'Train: {score_train:.4f}, valid: {score_valid:.4f}'
            print(msg.format(score_train=mean_score_train, score_valid=mean_score_valid))
        return {
            'loss': 1 - mean_score_valid,
            'status': STATUS_OK,
            'score': {'train': mean_score_train, 'valid': mean_score_valid}
        }

    def _save_experiments_results(
            self,
            trials: hyperopt.Trials,
            experiments_path: str,
            experiments_name: str
    ) -> None:
        """

        :param experiments_path:
        :param experiments_name:
        :return:
        """
        with open(os.path.join(experiments_path, '.'.join([experiments_name, 'hpopt'])), 'rb') as experiments:
            pickle.dump(trials, experiments)