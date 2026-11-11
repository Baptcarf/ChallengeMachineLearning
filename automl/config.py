import os
from pathlib import Path

TIME_BUDGET = int(os.getenv("AUTOML_TIME_BUDGET", "300"))
N_JOBS = int(os.getenv("AUTOML_N_JOBS", "1"))
N_TRIALS = int(os.getenv("AUTOML_N_TRIALS", "20"))

DATA_ROOT = Path("/info/corpus/ChallengeMachineLearning")
RANDOM_STATE = 42