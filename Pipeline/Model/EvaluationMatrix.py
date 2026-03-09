import math

import numpy as np
import pandas as pd
from scipy import stats


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator != 0 else 0.0

class EvaluationMatrix:
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true).ravel()
        self.y_pred = np.asarray(y_pred).ravel()
        self.n = len(self.y_true)

        self.TP = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.TN = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.FP = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.FN = np.sum((self.y_true == 1) & (self.y_pred == 0))

    def get_tp(self): return self.TP
    def get_tn(self): return self.TN
    def get_fp(self): return self.FP
    def get_fn(self): return self.FN

    def get_accuracy(self)      : return safe_divide(self.TP + self.TN, self.n)
    def get_precision(self)     : return safe_divide(self.TP, self.TP + self.FP)
    def get_recall(self)        : return safe_divide(self.TP, self.TP + self.FN)
    def get_npv(self)           : return safe_divide(self.TN, self.TN + self.FN)
    def get_specificity(self)   : return safe_divide(self.TN, self.TN + self.FP)
    def get_bal_accuracy(self)  : return (self.get_recall() + self.get_specificity()) / 2
    def get_f1_score(self)      :
        precision = self.get_precision()
        recall = self.get_recall()
        return safe_divide(2 * precision * recall, precision + recall)
    def get_f2_score(self)      :
        beta_sq = 4.0
        precision = self.get_precision()
        recall = self.get_recall()
        return safe_divide((1 + beta_sq) * precision * recall,
                                    (beta_sq * precision) + recall)
    def get_mcc(self)           :
        mcc_denominator = math.sqrt(
            (self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN))
        return safe_divide(self.TP * self.TN - self.FP * self.FN, mcc_denominator)

    def get_all_metrics(self):
        return {
            "Accuracy"          : self.get_accuracy(),
            "Precision"         : self.get_precision(),
            "Recall"            : self.get_recall(),
            "NPV"               : self.get_npv(),
            "Specificity"       : self.get_specificity(),
            "F1-Score"          : self.get_f1_score(),
            "F2-Score"          : self.get_f2_score(),
            "Balanced Accuracy" : self.get_bal_accuracy(),
            "MCC"               : self.get_mcc()
        }
    def get_report(self):
        metrics = self.get_all_metrics()
        rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}
        return {
            "Counts": {"TP": self.TP, "TN": self.TN, "FP": self.FP, "FN": self.FN},
            "Metrics": rounded_metrics
        }

    @staticmethod
    def combination_evaluation(data_frame):
        elm_seed_evaluation     = EvaluationMatrix.elm_seed_evaluation(data_frame)
        data_seed_evaluation    = EvaluationMatrix.data_seed_evaluation(data_frame)
        return elm_seed_evaluation,data_seed_evaluation
    @staticmethod
    def random_seed_metrics(data_frame, kappa=3.0):
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