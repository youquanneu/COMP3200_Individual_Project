import math

import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator != 0 else 0.0

class EvaluationMatrix:
    def __init__(self, y_true, y_pred):

        self.raw_y_true = np.asarray(y_true).ravel()
        self.raw_y_pred = np.asarray(y_pred).ravel()

        if self.raw_y_true.shape[0] == 0 | self.raw_y_pred.shape[0] == 0:
            raise ValueError("Input arrays cannot be empty.")
        elif self.raw_y_true.shape[0] != self.raw_y_pred.shape[0]:
            raise ValueError(f"Length mismatch: y_true({len(self.raw_y_true)}) != y_pred({len(self.raw_y_pred)})")

        self.classes = np.unique(np.concatenate((self.raw_y_true, self.raw_y_pred)))
        self.is_multiclass = ( len(np.unique(y_true)) > 2 ) | ( len(np.unique(y_pred)) > 2 )
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

    def _calc_metric(self, metric_func):
        if not self.is_multiclass:
            return metric_func(self.TP, self.TN, self.FP, self.FN)
        else:
            class_scores = [metric_func(m['TP'], m['TN'], m['FP'], m['FN']) for m in self.overall_metrics.values()]
            return float(np.mean(class_scores))

    def get_accuracy(self):
        return np.sum(self.y_true == self.y_pred) / self.n

    def get_precision(self):
        return self._calc_metric(lambda tp, tn, fp, fn: safe_divide(tp, tp + fp))

    def get_recall(self):
        return self._calc_metric(lambda tp, tn, fp, fn: safe_divide(tp, tp + fn))

    def get_npv(self):
        return self._calc_metric(lambda tp, tn, fp, fn: safe_divide(tn, tn + fn))

    def get_specificity(self):
        return self._calc_metric(lambda tp, tn, fp, fn: safe_divide(tn, tn + fp))

    def get_bal_accuracy(self):
        return (self.get_recall() + self.get_specificity()) / 2

    def get_f1_score(self):
        return self._calc_metric(lambda tp, tn, fp, fn: safe_divide(2 * tp, 2 * tp + fp + fn))

    def get_f2_score(self):
        beta_sq = 4.0
        return self._calc_metric(
            lambda tp, tn, fp, fn: safe_divide((1 + beta_sq) * tp, (1 + beta_sq) * tp + beta_sq * fn + fp)
        )

    def get_mcc(self):
        def mcc_calc(tp, tn, fp, fn):
            denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            return safe_divide(tp * tn - fp * fn, denominator)
        return self._calc_metric(mcc_calc)

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

    @staticmethod
    def combination_evaluation(data_frame):
        elm_seed_evaluation     = EvaluationMatrix.elm_seed_evaluation(data_frame)
        data_seed_evaluation    = EvaluationMatrix.data_seed_evaluation(data_frame)
        return elm_seed_evaluation,data_seed_evaluation
    @staticmethod
    def random_seed_metrics(data_frame):
        ignore_cols = ['Hidden_Nodes', 'Activation_Function', 'Lambda_Value','Data_Seed', 'Fold', 'ELM_Seed']
        metric_cols = [col for col in data_frame.columns if col not in ignore_cols]
        clean_df = data_frame[metric_cols]

        summary_df = clean_df.agg(['mean', 'std', 'min', 'max']).transpose()

        flat_results = {}
        for metric, row in summary_df.iterrows():
            flat_results[f"avg_{metric}_Seed_Mean"] = round(row['mean'], 4)
            flat_results[f"avg_{metric}_Seed_Std"] = round(row['std'], 4)
            flat_results[f"avg_{metric}_Seed_Min"] = round(row['min'], 4)
            flat_results[f"avg_{metric}_Seed_Max"] = round(row['max'], 4)

        final_row = pd.DataFrame([flat_results])
        return final_row

    @staticmethod
    def _aggregate_by_seed(data_frame, seed_col):
        """
        Core logic for seed evaluation.
        Consolidated to prevent logic drift between different seed types.
        """
        model_columns = ['Hidden_Nodes', 'Activation_Function', 'Lambda_Value']

        # Group by model columns and the dynamically passed seed column
        grouped_df = data_frame.groupby(model_columns + [seed_col]).mean(numeric_only=True).reset_index()

        # Note: We safely ignore ALL seed tracking columns regardless of which one we are grouping by
        ignore_cols = ['Data_Seed', 'Fold', 'ELM_Seed'] + model_columns
        metric_cols = [col for col in grouped_df.columns if col not in ignore_cols]

        summary_df = grouped_df.groupby(model_columns)[metric_cols].agg(['mean', 'std', 'min', 'max'])

        # Flatten the MultiIndex columns for clean pipeline extraction
        summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
        return summary_df.reset_index()

    @staticmethod
    def elm_seed_evaluation(data_frame):
        return EvaluationMatrix._aggregate_by_seed(data_frame, 'ELM_Seed')

    @staticmethod
    def data_seed_evaluation(data_frame):
        return EvaluationMatrix._aggregate_by_seed(data_frame, 'Data_Seed')