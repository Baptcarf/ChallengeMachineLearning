from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("/info/corpus/ChallengeMachineLearning")
Task = Literal["regression", "binary", "multiclass", "multilabel"]


@dataclass
class Dataset:
    path: Path
    data: np.ndarray | sparse.csr_matrix
    types: list[str]
    solution: np.ndarray | None  # La solution est nul quand on "fit"
    isSparse: bool
    tache: Task | None = field(init=False)

    X_train: np.ndarray = field(init=False)
    X_test: np.ndarray = field(init=False)
    y_train: np.ndarray = field(init=False)
    y_test: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.solution is not None:
            self.task = detecterTache(self.solution)

    @property
    def n_cols(self) -> int:
        return self.solution.shape[1]

    def __repr__(self) -> str:
        return (
            f"Dataset(\n"
            f"name={self.path.name!r},\n"
            f"path={str(self.path)!r},\n"
            f"format={'sparse' if self.isSparse else 'dense'},\n"
            f"samples={self.data.shape[0]},\n"
            f"features={self.data.shape[1]},\n"
            f"solution_shape={self.solution.shape})"
        )

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.solution.ravel(), test_size=0.2, random_state=42
        )
        return (self.train(), self.test())

    def train(self) -> np.ndarray:
        return (self.X_train, self.y_train)

    def test(self) -> np.ndarray:
        return (self.X_test, self.y_test)


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
    solution = path / f"{prefix}.solution"

    types = [line.strip() for line in types.read_text().splitlines() if line.strip()]
    solution = np.loadtxt(solution, dtype=float, ndmin=2)

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
