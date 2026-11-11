import logging
import sys

import optuna

VERT = "\033[32m"
RESET = "\033[0m"


def configurerLogs(niveau: int = logging.INFO, fichier: str | None = None, optunaLogs: bool = True) -> None:
    """À appeler UNE fois"""
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(f"{VERT}%(asctime)s{RESET} | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    handlers: list[logging.Handler] = [console]

    if fichier:
        # sans couleur dans le fichier
        handlerFichier = logging.FileHandler(fichier, encoding="utf-8")
        handlerFichier.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        handlers.append(handlerFichier)

    logging.basicConfig(level=logging.WARNING, handlers=handlers, force=True)
    logging.getLogger("automl").setLevel(niveau)

    if optunaLogs:
        optuna.logging.disable_default_handler()  # évite les doublons
        optuna.logging.enable_propagation()  # passe par console + fichier
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
