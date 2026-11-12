"""
Prétraitement automatique des données pour le projet AutoML.

- Gestion des valeurs manquantes
- Valeurs infinies (+inf et -inf)
- Colonnes entièrement vides (100 % NaN)
- Nombre de colonnes incohérent entre train et test
- Traitement des variables numériques et catégorielles
- Gestion des matrices sparse
- Catégories inconnues lors de la prédiction
- Compatibilité avec fit() et predict()
- Données sans labels lors de predict()
"""
