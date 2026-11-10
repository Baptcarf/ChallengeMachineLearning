from .data import *
from .optimisation import *
from optuna import create_study

class AutoML:

    def __init__(self):
        self.allData  = ""

    def fit(self,dataDest):
        self.allData =  listerDatasets(prefix=dataDest)


        '''
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
        '''

        data = lireDataset(self.allData[5])
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
            n_trials=5
        )

        print("Meilleur score :", study.best_value)
        print("Meilleurs paramètres :", study.best_params)


    def eval(self):
        return 

    
    def predict(self,test_path):
        self.test_path = test_path

    def print(self):
        print(self.dataDest)

