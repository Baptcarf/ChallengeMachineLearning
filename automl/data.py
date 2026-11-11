from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

from .config import DATA_ROOT, RANDOM_STATE

Task = Literal["regression", "binary", "multiclass", "multilabel"]


@dataclass
class Split:
    x: np.ndarray | sparse.csr_matrix
    y: np.ndarray


@dataclass
class Dataset:
    path: Path
    data: np.ndarray | sparse.csr_matrix
    types: list[str]
    solution: np.ndarray | None  # La solution est nul quand on "fit"
    isSparse: bool
    tache: Task | None = field(init=False, default=None)

    train: Split | None = field(init=False, default=None)  # 70% train
    valid: Split | None = field(init=False, default=None)  # 10% Validation
    test: Split | None = field(init=False, default=None)  # 20% Test

    def __post_init__(self) -> None:
        if self.solution is not None:
            self._detecterTache()

    def _detecterTache(self) -> None:
        y = self.solution
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if not np.isin(y, (0.0, 1.0)).all():  # Une valeur qui n'est pas 0 ou 1 => régression
            self.tache = "regression"
        if y.shape[1] == 1:  # Une seule colonne de 0/1 =W binaire
            self.tache = "binary"
        if (y.sum(axis=1) == 1).all():  # Plusieurs colonnes, exactement un 1 par ligne => Multiclass
            self.tache = "multiclass"
        self.tache = "multilabel"  # Sinon (plusieurs 1 possibles, ou des lignes à zéro) => multilabel.

    def _labels_stratif(self, solution: np.ndarray | None = None) -> np.ndarray | None:
        """
        Prépare les étiquettes à utiliser pour répartir les données
        en conservant autant que possible les proportions des classes

        Selon le type de problème, les étiquettes ne sont pas organisées
        de la même manière. On les transforme donc en un tableau simple
        contenant une seule valeur par observation.

        Retourne None si la stratification n'est pas possible ou nécessaire.
        """
        # Utilise les solutions fournies en paramètre, ou celles du dataset
        # si aucune solution n'est précisée
        y = self.solution if solution is None else solution

        # Sans solution, il est impossible de connaître les classes
        # auxquelles appartiennent les observations
        if y is None:
            return None

        # Classification binaire : chaque observation possède déjà
        # une seule étiquette (0 ou 1). On transforme le tableau en 1D
        if self.tache == "binary":
            return y.ravel()

        # Classification multiclasses : chaque observation possède
        # une seule classe, représentée par un vecteur contenant un seul 1.
        # On récupère la position de ce 1 pour obtenir son numéro de classe.
        if self.tache == "multiclass":
            return y.argmax(axis=1)

        # Classification multilabel : une observation peut posséder
        # plusieurs étiquettes simultanément
        if self.tache == "multilabel":
            # Chaque combinaison différente d'étiquettes est considérée
            # comme une classe distincte afin de pouvoir effectuer
            # une stratification classique
            #
            # Exemple :
            # [1, 0, 1] devient une classe
            # [0, 1, 0] devient une autre classe
            # Les combinaisons identiques reçoivent le même numéro
            labels = np.unique(y, axis=0, return_inverse=True)[1]

            # Chaque classe doit avoir au moins deux observations
            # pour permettre une séparation stratifiée
            # Si ce n'est pas le cas, on désactive la stratification
            if np.bincount(labels).min() < 2:
                return None
            return labels

        # En régression, les valeurs à prédire sont continues
        # il n'existe pas de classes à conserver dans les mêmes proportions.
        return None

    def split(self, valid_size: float = 0.1, test_size: float = 0.2, random_state: int = RANDOM_STATE) -> None:
        """
        Sépare le dataset en trois ensembles :
        - Train : utilisé pour entraîner le modèle
        - Validation : utilisé pour évaluer et ajuster le modèle pendant son développement
        - Test : utilisé pour évaluer les performances finales du modèle

        Les proportions sont définies par valid_size et test_size.
        La stratification est utilisée lorsque cela est possible afin de
        conserver les proportions des classes dans chaque ensemble.
        """
        if self.solution is None:
            raise ValueError("Impossible de séparer un dataset sans solution")

        # On sépare le dataset en deux parties :
        # - 80 % pour l'entraînement et la validation
        # - 20 % pour le test
        #
        # random_state permet de reproduire exactement la même séparation
        # à chaque exécution
        #
        # stratify permet de conserver autant que possible les mêmes
        # proportions de classes dans les deux ensembles
        xTrainValid, xTest, yTrainValid, yTest = train_test_split(
            self.data, self.solution, test_size=test_size, random_state=random_state, stratify=self._labels_stratif()
        )

        # On calcule la proportion de validation à appliquer sur les données
        # restantes, car valid_size représente une proportion du dataset INITIAL
        #
        # Exemple :
        # - 20 % ont déjà été réservés pour le test.
        # - Il reste donc 80 % du dataset.
        # - Pour obtenir 10 % de validation au total :
        #   10 / 80 = 12,5 % des données restantes
        relativeValideSize = valid_size / (1 - test_size)
        xTrain, xValid, yTrain, yValid = train_test_split(
            xTrainValid,
            yTrainValid,
            test_size=relativeValideSize,
            random_state=random_state,
            stratify=self._labels_stratif(yTrainValid),
        )

        self.train = Split(xTrain, yTrain)
        self.valid = Split(xValid, yValid)
        self.test = Split(xTest, yTest)

    def __repr__(self) -> str:
        return (
            f"Dataset(\n"
            f"name={self.path.name!r},\n"
            f"path={str(self.path)!r},\n"
            f"format={'sparse' if self.isSparse else 'dense'},\n"
            f"samples={self.data.shape[0]},\n"
            f"features={self.data.shape[1]},\n"
            f"solution_shape={None if self.solution is None else self.solution.shape}),\n"
            f"tache={self.tache})"
        )


def listerDatasets(root: Path = DATA_ROOT, prefix: str = "data") -> list[Path]:
    return sorted(
        dataSetDossier for dataSetDossier in root.iterdir() if dataSetDossier.is_dir() and dataSetDossier.name.startswith(prefix)
    )


def lireDataset(path: Path, n_features: int | None = None) -> Dataset:
    prefix = path.name

    data = path / f"{prefix}.data"
    types = path / f"{prefix}.type"
    solutionPath = path / f"{prefix}.solution"

    types = [line.strip() for line in types.read_text().splitlines() if line.strip()]
    solution = np.loadtxt(solutionPath, dtype=float, ndmin=2) if solutionPath.exists() else None

    with data.open() as file:
        premiereLigne = file.readline()

    dataIsSparse = ":" in premiereLigne
    if not dataIsSparse:
        data = np.loadtxt(data, dtype=float, ndmin=2)
        return Dataset(path, data, types, solution, dataIsSparse)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    nLignes = 0

    with data.open() as file:
        for row, line in enumerate(file):
            nLignes += 1

            for token in line.split():
                column, value = token.split(":", 1)

                rows.append(row)
                columns.append(int(column) - 1)
                values.append(float(value))

    if n_features is None:
        n_features = len(types)

    data = sparse.csr_matrix((values, (rows, columns)), shape=(nLignes, n_features))
    return Dataset(path, data, types, solution, dataIsSparse)
