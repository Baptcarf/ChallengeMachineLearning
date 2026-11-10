
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from optuna import create_study
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from typing import Protocol
import numpy as np


class Objectif(Protocol):

    def optimisation(self,trial,X,y):
        ...



class Regression(Objectif):
    def optimisation(self,trial,X,y):
        alpha = trial.suggest_float(
            "alpha", 1e-4, 1e4, log=True
        )
        ridge = Ridge()
        ridge.set_params(alpha=alpha)

        scores = cross_val_score(
            ridge,
            X,
            y,
            cv=4,
            scoring="r2",
            n_jobs=32
        )

        return scores.mean()

class Binaire(Objectif):
    def optimisation(self,trial, X, y):
        max_depth = trial.suggest_int("max_depth", 2, 32)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        randomForest = RandomForestClassifier()
        randomForest.set_params(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        scores = cross_val_score(
            randomForest,
            X,
            y,
            cv=4,
            scoring="f1_weighted",
            n_jobs=32
        )

        return scores.mean()

class MultiClass(Objectif):
    def optimisation(self, trial, X, y):

        C = trial.suggest_float("C", 1e-3, 10, log=True)

        solver = trial.suggest_categorical(
            "solver", ["lbfgs", "saga"]
        )

        l1_ratio = (
            trial.suggest_float("l1_ratio", 0, 1)
            if solver == "saga"
            else 0
        )

        max_iter = trial.suggest_int("max_iter", 100, 500, step=100)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        logisticRegression = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            l1_ratio=l1_ratio
        )

        scores = cross_val_score(
            logisticRegression,
            X,
            y,
            cv=4,
            scoring="f1_weighted",
            n_jobs=32
        )

        return scores.mean()

class MultiLabel(Objectif):
    def optimisation(self,trial, X, y):
        max_depth = trial.suggest_int("max_depth", 2, 32)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        randomForest = RandomForestClassifier()
        randomForest.set_params(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        scores = cross_val_score(
            randomForest,
            X,
            y,
            cv=4,
            scoring="f1_weighted",
            n_jobs=32
        )

        return scores.mean()

        

