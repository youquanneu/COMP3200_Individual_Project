import math

import numpy as np


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator != 0 else 0.0

class EvaluationMatrix:
    def __init__(self, y_true, y_pred):

        self.raw_y_true = np.asarray(y_true).ravel()
        self.raw_y_pred = np.asarray(y_pred).ravel()

        if self.raw_y_true.shape[0] == 0 or self.raw_y_pred.shape[0] == 0:
            raise ValueError("Input arrays cannot be empty.")

        elif self.raw_y_true.shape[0] != self.raw_y_pred.shape[0]:
            raise ValueError(f"Length mismatch: y_true({len(self.raw_y_true)}) != y_pred({len(self.raw_y_pred)})")

        self.classes = np.unique(np.concatenate([self.raw_y_true, self.raw_y_pred]))
        self.is_multiclass = ( len(np.unique(y_true)) > 2 ) or ( len(np.unique(y_pred)) > 2 )
        self.n = len(self.raw_y_true)

        if not self.is_multiclass:
            self.y_true = np.where(self.raw_y_true == np.min(self.raw_y_true), -1, 1)
            self.y_pred = np.where(self.raw_y_pred == np.min(self.raw_y_pred), -1, 1)

            self.TP = np.sum((self.y_true ==  1) & (self.y_pred ==  1))
            self.TN = np.sum((self.y_true == -1) & (self.y_pred == -1))
            self.FP = np.sum((self.y_true == -1) & (self.y_pred ==  1))
            self.FN = np.sum((self.y_true ==  1) & (self.y_pred == -1))

        else:
            self.y_true = self.raw_y_true
            self.y_pred = self.raw_y_pred
            self.overall_metrics = {}

            for c in self.classes:
                tp = np.sum((self.y_true == c) & (self.y_pred == c))
                tn = np.sum((self.y_true != c) & (self.y_pred != c))
                fp = np.sum((self.y_true != c) & (self.y_pred == c))
                fn = np.sum((self.y_true == c) & (self.y_pred != c))
                self.overall_metrics[c] = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

    def calculate_metric(self, metric_func):
        if not self.is_multiclass:
            return metric_func(self.TP, self.TN, self.FP, self.FN)
        else:
            class_scores = [metric_func(m['TP'], m['TN'], m['FP'], m['FN']) for m in self.overall_metrics.values()]
            return float(np.mean(class_scores))

    def get_accuracy(self):
        return np.sum(self.y_true == self.y_pred) / self.n

    def get_precision(self):
        return self.calculate_metric(lambda tp, tn, fp, fn: safe_divide(tp, tp + fp))

    def get_recall(self):
        return self.calculate_metric(lambda tp, tn, fp, fn: safe_divide(tp, tp + fn))

    def get_npv(self):
        return self.calculate_metric(lambda tp, tn, fp, fn: safe_divide(tn, tn + fn))

    def get_specificity(self):
        return self.calculate_metric(lambda tp, tn, fp, fn: safe_divide(tn, tn + fp))

    def get_bal_accuracy(self):
        return (self.get_recall() + self.get_specificity()) / 2

    def get_f1_score(self):
        return self.calculate_metric(lambda tp, tn, fp, fn: safe_divide(2 * tp, 2 * tp + fp + fn))

    def get_f2_score(self):
        beta_sq = 4.0
        return self.calculate_metric(
            lambda tp, tn, fp, fn: safe_divide((1 + beta_sq) * tp, (1 + beta_sq) * tp + beta_sq * fn + fp)
        )

    def get_mcc(self):
        def mcc_calc(tp, tn, fp, fn):
            denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            return safe_divide(tp * tn - fp * fn, denominator)
        return self.calculate_metric(mcc_calc)

    def get_all_metrics(self):
        return {
            "Accuracy"      : self.get_accuracy(),
            "Precision"     : self.get_precision(),
            "Recall"        : self.get_recall(),
            "NPV"           : self.get_npv(),
            "Specificity"   : self.get_specificity(),
            "F1-Score"      : self.get_f1_score(),
            "F2-Score"      : self.get_f2_score(),
            "Bal Accuracy"  : self.get_bal_accuracy(),
            "MCC"           : self.get_mcc()
        }

    def get_report(self):
        metrics = self.get_all_metrics()
        rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}

        if not self.is_multiclass:
            counts = {"TP": self.TP, "TN": self.TN, "FP": self.FP, "FN": self.FN}
        else:
            counts = {
                "Total_Correct"     : int(np.sum(self.y_true == self.y_pred)),
                "Total_Incorrect"   : int(np.sum(self.y_true != self.y_pred)),
                "Unique_Classes"    : len(self.classes)
            }

        return {
            "Counts": counts,
            "Metrics": rounded_metrics
        }