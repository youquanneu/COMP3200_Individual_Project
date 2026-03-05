import math

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0

class EvaluationMatrix:
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true).ravel()
        self.y_pred = np.asarray(y_pred).ravel()
        self.n = len(self.y_true)

        self.TP = sum(1 for a, p in zip(self.y_true, self.y_pred) if a == 1 and p == 1)
        self.TN = sum(1 for a, p in zip(self.y_true, self.y_pred) if a == 0 and p == 0)
        self.FP = sum(1 for a, p in zip(self.y_true, self.y_pred) if a == 0 and p == 1)
        self.FN = sum(1 for a, p in zip(self.y_true, self.y_pred) if a == 1 and p == 0)

        self.accuracy   = safe_divide(self.TP + self.TN, self.n)
        self.precision  = safe_divide(self.TP, self.TP + self.FP)
        self.recall     = safe_divide(self.TP, self.TP + self.FN)
        self.npv        = safe_divide(self.TN, self.TN + self.FN)
        self.specificity= safe_divide(self.TN, self.TN + self.FP)

        self.f1_score   = safe_divide(2 * self.precision * self.recall, self.precision + self.recall)
        self.bal_accuracy = (self.recall + self.specificity) / 2
        mcc_denominator = math.sqrt((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN))
        self.mcc        = safe_divide(self.TP * self.TN - self.FP * self.FN, mcc_denominator)

    def get_accuracy(self)      : return self.accuracy
    def get_precision(self)     : return self.precision
    def get_recall(self)        : return self.recall
    def get_npv(self)           : return self.npv
    def get_specificity(self)   : return self.specificity
    def get_f1_score(self)      : return self.f1_score
    def get_bal_accuracy(self)  : return self.bal_accuracy
    def get_mcc(self)           : return self.mcc

    def get_all_metrics(self):
        return {
            "Accuracy"          : self.get_accuracy(),
            "Precision"         : self.get_precision(),
            "Recall"            : self.get_recall(),
            "NPV"               : self.get_npv(),
            "Specificity"       : self.get_specificity(),
            "F1-Score"          : self.get_f1_score(),
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
    def k_fold_metrics(dataframe,activation_function, hidden_size,regularization_lambda=0.0):

        summary = dataframe.agg(['mean', 'std', 'min', 'max']).transpose()

        final_dict = {}
        for metric, row in summary.iterrows():
            final_dict[f"avg_{metric}"] = round(row['mean'], 4)
            final_dict[f"std_{metric}"] = round(row['std'], 4)
            final_dict[f"min_{metric}"] = round(row['min'], 4)
            final_dict[f"max_{metric}"] = round(row['max'], 4)

        final_row = pd.DataFrame([final_dict])
        final_row.insert(0, 'Hidden_Nodes', hidden_size)

        if hasattr(activation_function, '__name__'):
            final_row.insert(1, 'Activation', activation_function.__name__)
        else:
            final_row.insert(1, 'Activation', 'Unknown')

        final_row.insert(2,'Lambda_Value',regularization_lambda)

        return final_row
    @staticmethod
    def random_seed_metrics(data_frame):
        avg_cols = [col for col in data_frame.columns if col.startswith('avg_')]
        clean_df = data_frame[avg_cols]

        summary_df = clean_df.agg(['mean', 'std', 'min', 'max']).transpose()

        flat_results = {}
        for metric, row in summary_df.iterrows():
            flat_results[f"{metric}_Seed_Mean"] = round(row['mean'], 4)
            flat_results[f"{metric}_Seed_Std"]  = round(row['std'], 4)
            flat_results[f"{metric}_Seed_Min"]  = round(row['min'], 4)
            flat_results[f"{metric}_Seed_Max"]  = round(row['max'], 4)

        final_row = pd.DataFrame([flat_results])
        hidden_size = data_frame['Hidden_Nodes'].iloc[0]
        activation = data_frame['Activation'].iloc[0]
        lam_val = data_frame['Lambda_Value'].iloc[0]

        final_row.insert(0, 'Hidden_Nodes', hidden_size)
        final_row.insert(1, 'Activation', activation)
        final_row.insert(2, 'Lambda_Value', lam_val)

        return final_row