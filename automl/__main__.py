from optuna import create_study

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
        """
        for d in self.allData:
            data = lireDataset(d)
            study = create_study(direction="maximize")
            if data.tache == "regression":
                optimiseur = Regression()
            elif data.tache == "binary":
                optimiseur = Binaire()
            elif data.tache == "multiclass":
                optimiseur = MultiClass()
            else:
                optimiseur = MultiLabel()
            data.split()
            study.optimize(
                lambda trial: optimiseur.optimisation(trial, data.train.x,data.train.y),
                n_trials=30
            )

            print("Meilleur score :", study.best_value)
            print("Meilleurs paramètres :", study.best_params)
        """

        data = lireDataset(dataDest)
        study = create_study(direction="maximize")
        if data.tache == "regression":
            optimiseur = Regression()
        elif data.tache == "binary":
            optimiseur = Binaire()
        elif data.tache == "multiclass":
            optimiseur = MultiClass()
        else:
            optimiseur = MultiLabel()
        data.split()
        study.optimize(lambda trial: optimiseur.optimisation(trial, data.train.x, data.train.y), n_trials=5)

        print("Meilleur score :", study.best_value)
        print("Meilleurs paramètres :", study.best_params)

    def eval(self):
        return

    def predict(self, test_path):
        self.test_path = test_path