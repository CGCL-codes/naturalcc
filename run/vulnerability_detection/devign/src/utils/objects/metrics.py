import pandas as pd

from .. import log as logger
from sklearn.metrics import confusion_matrix
from sklearn import metrics


class Metrics:
    def __init__(self, outs, labels):
        self.scores = outs
        self.labels = labels
        self.transform()
        print(self.predicts)

    def transform(self):
        self.series = pd.Series(self.scores)
        self.predicts = self.series.apply(lambda x: 1 if x >= 0.5 else 0)
        self.predicts.reset_index(drop=True, inplace=True)

    def __str__(self):
        confusion = confusion_matrix(y_true=self.labels, y_pred=self.predicts)
        tn, fp, fn, tp = confusion.ravel()
        string = f"\nConfusion matrix: \n"
        string += f"{confusion}\n"
        string += f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
        string += '\n'.join([name + ": " + str(metric) for name, metric in self().items()])
        return string

    def __call__(self):
        _metrics = {"Accuracy": metrics.accuracy_score(y_true=self.labels, y_pred=self.predicts),
                    "Precision": metrics.precision_score(y_true=self.labels, y_pred=self.predicts),
                    "Recall": metrics.recall_score(y_true=self.labels, y_pred=self.predicts),
                    "F-measure": metrics.f1_score(y_true=self.labels, y_pred=self.predicts),
                    "Precision-Recall AUC": metrics.average_precision_score(y_true=self.labels, y_score=self.scores),
                    "AUC": metrics.roc_auc_score(y_true=self.labels, y_score=self.scores),
                    "MCC": metrics.matthews_corrcoef(y_true=self.labels, y_pred=self.predicts),
                    "Error": self.error()}

        return _metrics

    def log(self):
        excluded = ["Precision-Recall AUC", "AUC"]
        _metrics = self()
        msg = ' - '.join(
            [f"({name[:3]} {round(metric, 3)})" for name, metric in _metrics.items() if name not in excluded])

        logger.log_info('metrics', msg)

    def error(self):
        errors = [(abs(score - (1 if score >= 0.5 else 0))/(score+1e-8))*100 for score, label in zip(self.scores, self.labels)]

        return sum(errors)/len(errors)
