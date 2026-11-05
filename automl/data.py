from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA_ROOT = Path("/info/corpus/ChallengeMachineLearning")


@dataclass
class Dataset:
    path: Path
    data: np.ndarray | sparse.csr_matrix
    types: list[str]
    solution: np.ndarray
    isSparse: bool

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
    solution = pd.read_csv(solution, header=None, sep=r"\s+", engine="c").to_numpy()

    with data.open() as file:
        premiereLigne = file.readline()

    dataIsSparse = ":" in premiereLigne
    if not dataIsSparse:
        data = pd.read_csv(data, header=None, sep=r"\s+", engine="c", memory_map=True)
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
