from optuna import create_study
from optuna.samplers import TPESampler

from .config import RANDOM_STATE
from .data import *
from .optimisation import *


class AutoML:
    def __init__(self):
        self.dataset = None
        self.preprocessor = None
        self.model = None
        self.tache = None
        self.metrics = None

    def fit(self, dataDest) -> None:
        data = lireDataset(dataDest)
        data.split()
        sampler = TPESampler(seed=RANDOM_STATE)
        study = create_study(sampler=sampler, direction="maximize")
        if data.tache == "regression":
            optimiseur = Regression()
        elif data.tache == "binary":
            optimiseur = Binaire()
        elif data.tache == "multiclass":
            optimiseur = MultiClass()
        else:
            optimiseur = MultiLabel()
        study.optimize(lambda trial: optimiseur.optimisation(trial, data.train.x, data.train.y), n_trials=5)

        print("Meilleur score :", study.best_value)
        print("Meilleurs paramètres :", study.best_params)

    def eval(self):
        return

    def predict(self, test_path):
        self.test_path = test_path
