import automl

data_dest = "/info/corpus/ChallengeMachineLearning/data_A"
automl.fit(data_dest)  # entraine, compare les modeles, optimise
automl.eval()  # metriques de validation
# preds = automl.predict("/path/to/test")
