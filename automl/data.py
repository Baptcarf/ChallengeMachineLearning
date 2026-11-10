from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("/info/corpus/ChallengeMachineLearning")
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

    train: Split | None = field(init=False, default=None)
    valid: Split | None = field(init=False, default=None)
    test: Split | None = field(init=False, default=None)

    def __post_init__(self):
        if self.solution is not None:
            self.tache = detecterTache(self.solution)

    def _labels_stratif(self, solution: np.ndarray | None = None) -> np.ndarray | None:
        """Vecteur 1D de labels pour stratify, ou None si pas de stratification"""
        y = self.solution if solution is None else solution
        if y is None:
            return None

        if self.tache == "binary":
            return y.ravel()
        if self.tache == "multiclass":
            return y.argmax(axis=1)
        if self.tache == "multilabel":
            # chaque combinaison de labels distincte = une "classe"
            labels = np.unique(y, axis=0, return_inverse=True)[1]
            if np.bincount(labels).min() < 2:
                return None
            return labels
        return None  # regression

    def split(self, valid_size: float = 0.1, test_size: float = 0.2, random_state: int = 42):
        if self.solution is None:
            raise ValueError("Impossible de séparer un dataset sans solution")

        x_train_valid, x_test, y_train_valid, y_test = train_test_split(
            self.data, self.solution, test_size=test_size, random_state=random_state, stratify=self._labels_stratif()
        )

        relative_valid_size = valid_size / (1 - test_size)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_valid,
            y_train_valid,
            test_size=relative_valid_size,
            random_state=random_state,
            stratify=self._labels_stratif(y_train_valid),
        )

        self.train = Split(x_train, y_train)
        self.valid = Split(x_valid, y_valid)
        self.test = Split(x_test, y_test)

    def __repr__(self) -> str:
        return (
            f"Dataset(\n"
            f"name={self.path.name!r},\n"
            f"path={str(self.path)!r},\n"
            f"format={'sparse' if self.isSparse else 'dense'},\n"
            f"samples={self.data.shape[0]},\n"
            f"features={self.data.shape[1]},\n"
            f"solution_shape={None if self.solution is None else self.solution.shape})"
        )


def detecterTache(solution: np.ndarray) -> Task:
    y = np.asarray(solution)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if not np.isin(y, (0.0, 1.0)).all():  # Une valeur qui n'est pas 0 ou 1 => régression
        return "regression"
    if y.shape[1] == 1:  # Une seule colonne de 0/1 =W binaire
        return "binary"
    if (y.sum(axis=1) == 1).all():  # Plusieurs colonnes, exactement un 1 par ligne => Multiclass
        return "multiclass"
    return "multilabel"  # Sinon (plusieurs 1 possibles, ou des lignes à zéro) => multilabel.


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
