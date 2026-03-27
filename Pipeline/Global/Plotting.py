import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from Pipeline.Global.GlobalSetting import GlobalSetting


class Plotting:
    seaborn_theme = {
        "style"         : "whitegrid",
        "context"       : "paper",
        "font_scale"    : 1.2
    }

    rc_params_standard = {
        'font.family'       : 'sans-serif',
        'font.sans-serif'   : ['Arial', 'DejaVu Sans'],
        'figure.titlesize'  : 16,
        'axes.titlesize'    : 16,
        'axes.labelsize'    : 12,
        'axes.titleweight'  : 'bold',
        'axes.labelweight'  : 'bold',
        'legend.frameon'    : False,
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        'axes.axisbelow'    : True
    }

    @classmethod
    @contextmanager
    def _style_context(cls):
        """Yields a unified styling context for all plots."""
        sns.set_theme(**cls.seaborn_theme)
        with plt.rc_context(cls.rc_params_standard):
            yield

    @classmethod
    def _format_standard_axes(cls, ax, title=None, xlabel=None, ylabel=None):
        """Applies standard commercial styling to an axis."""
        if title: ax.set_title(title, pad=15)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Only draw legend if labels exist
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

    @classmethod
    def _get_metric(cls, metric: str) -> str:
        """Helper to resolve the fitness metric."""
        return metric if metric is not None else GlobalSetting.evaluation_function

    @classmethod
    def _save_figure(cls, fig, prefix: str, experiment_name: str, fitness_metric: str):
        """Internal helper to isolate I/O operations from plotting logic."""
        target_dir = GlobalSetting.get_figure_dir()
        safe_filename = experiment_name.replace(" ", "_").replace(":", "").replace("/", "-")
        file_path = os.path.join(target_dir, f"{prefix}_{safe_filename}_{fitness_metric}.png")

        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[I/O Trace] Figure exported successfully: {file_path}")

    @classmethod
    def plot_abc_dashboard(cls, convergence_df: pd.DataFrame, scout_df: pd.DataFrame,
                           experiment_name: str, results_df: pd.DataFrame = None,
                           fitness_metric: str = None, is_final_record=False):

        metric = cls._get_metric(fitness_metric)

        with cls._style_context():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16),
                                           gridspec_kw={'height_ratios': [6, 1]},
                                           facecolor='white')
            fig.suptitle(f'ABC-ELM Telemetry | {experiment_name}', y=0.98, fontsize=18, fontweight='bold')
            iterations = np.arange(1, len(convergence_df) + 1)

            # Ax1: Convergence
            for seed_col in convergence_df.columns:
                ax1.plot(iterations, convergence_df[seed_col], linestyle=':', linewidth=1, alpha=0.75)
            ax1.plot(iterations, convergence_df.mean(axis=1), color='#D62728', linewidth=3.5,
                     label='Average Convergence')

            if results_df is not None and metric in results_df.columns:
                test_mean, test_std = results_df[metric].mean(), results_df[metric].std()
                ax1.axhline(y=test_mean, color='#2ca02c', linestyle='--', linewidth=2.5,
                            label=f'Final Test Mean ({test_mean:.4f})')
                ax1.axhspan(ymin=test_mean - test_std, ymax=test_mean + test_std, color='#2ca02c', alpha=0.15,
                            label=f'Test Variance (± 1σ, σ={test_std:.4f})')
                ax1.text(x=iterations[-1], y=test_mean + test_std, s=f'+1 σ bound ({test_mean + test_std:.3f})',
                         color='#1b5e20', va='bottom', ha='right', fontweight='bold', fontsize=11)
                ax1.text(x=iterations[-1], y=test_mean - test_std, s=f'-1 σ bound ({test_mean - test_std:.3f})',
                         color='#1b5e20', va='top', ha='right', fontweight='bold', fontsize=11)

            ax1.set_ylim(top=1.0)
            ax1.grid(axis='x', visible=False)  # Override standard x-grid
            cls._format_standard_axes(ax1, title='Phase 1: Convergence Trajectory', ylabel=f'Fitness ({metric})')

            # Ax2: Scouts
            ax2.bar(iterations, scout_df.sum(axis=1), color='#1f77b4', alpha=0.8, edgecolor='black')
            cls._format_standard_axes(ax2, title='Phase 2: Scout Bee Interventions', xlabel='Iteration / Bee Cycle',
                                      ylabel='Total Triggers')

            plt.tight_layout()
            fig.subplots_adjust(top=0.90, hspace=0.35)

            if is_final_record:
                cls._save_figure(fig, "ABC_Telemetry", experiment_name, metric)
            plt.show()

    @classmethod
    def plot_cv_grid(cls,
                     all_train_curves   : list,
                     all_val_curves     : list,
                     all_test_scores    : list,
                     experiment_name    : str,
                     fitness_metric     : str = None,
                     is_final_record    : bool = False,
                     save_fold_record   : bool = False):
        """
        Orchestrates the plotting of individual CV folds and their aggregate.
        Delegates rendering to isolated private methods.
        """

        if fitness_metric is None:
            fitness_metric = GlobalSetting.evaluation_function

        total_folds = len(all_train_curves)

        with cls._style_context():
            # 1. Plot Individual Folds
            for idx in range(len(all_train_curves)):
                cls._plot_single_fold(
                    all_train_curves[idx] ,
                    all_val_curves[idx],
                    all_test_scores[idx],
                    idx,
                    total_folds,
                    experiment_name,
                    fitness_metric,
                    is_final_record,
                    save_fold_record
                )

            # 2. Plot Aggregate Mean ± STD
            cls._plot_cv_aggregate(
                all_train_curves,
                all_val_curves,
                all_test_scores,
                total_folds,
                experiment_name,
                fitness_metric,
                is_final_record
            )

    @classmethod
    def _plot_single_fold(cls,
                          train_curve,
                          val_curve,
                          test_score,
                          fold_idx,
                          total_folds,
                          experiment_name,
                          fitness_metric,
                          is_final_record,
                          save_fold_record):
        """Internal helper to render and save a single CV fold figure."""
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        iterations = np.arange(1, len(train_curve) + 1)

        final_train_fitness = train_curve[-1]
        ax.plot(iterations, np.array(train_curve), color='#1f77b4', linewidth=2.5, label=f'Best Fitness ({final_train_fitness:.5f})')

        if len(val_curve) > 0:
            final_val_fitness = val_curve[-1]
            ax.plot(iterations, np.array(val_curve), color='#ff7f0e', linewidth=2.5, label=f'Val Fitness ({final_val_fitness:.5f})')

        if test_score is not None:
            ax.axhline(y = test_score, color='#2ca02c', linestyle='--', linewidth=2.5, label=f'Test ({test_score:.5f})')

        ax.set_title(f'{experiment_name} | Trace : Fold {fold_idx + 1}/{total_folds}', pad=15, fontsize=16, fontweight='bold')
        ax.set_xlabel('Iterations', fontweight='bold')
        ax.set_ylabel(f'Fitness ({fitness_metric})', fontweight='bold')

        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

        plt.tight_layout()

        if is_final_record & save_fold_record:
            cls._save_figure(fig, f"Fold_{fold_idx + 1}", experiment_name, fitness_metric)
        plt.show()

    @classmethod
    def _plot_cv_aggregate(cls,
                           train_curves,
                           val_curves,
                           test_scores,
                           total_folds,
                           experiment_name,
                           fitness_metric,
                           is_final_record):
        """Internal helper to calculate statistics and render the aggregate summary figure."""
        fig, agg_ax = plt.subplots(figsize=(12, 7), facecolor='white')

        # Matrix conversions
        train_matrix, val_matrix = np.array(train_curves), np.array(val_curves)

        # Statistics
        mean_train, std_train = np.mean(train_matrix, axis=0), np.std(train_matrix, axis=0)
        mean_val, std_val = np.mean(val_matrix, axis=0), np.std(val_matrix, axis=0)

        valid_scores = [s for s in test_scores if s is not None]
        mean_test = np.mean(valid_scores) if valid_scores else 0
        std_test = np.std(valid_scores) if valid_scores else 0

        iterations = np.arange(1, len(mean_train) + 1)

        final_mean_train = mean_train[-1]
        final_std_train = std_train[-1]
        final_mean_val = mean_val[-1]
        final_std_val = std_val[-1]
        # Rendering
        agg_ax.plot(iterations, mean_train, label=f'Best Fitness ({final_mean_train:.5f} ± {final_std_train:.5f})', color='#1f77b4', linewidth=3.0)
        agg_ax.fill_between(iterations, mean_train - std_train, mean_train + std_train, color='#1f77b4', alpha=0.2)

        agg_ax.plot(iterations, mean_val, label=f'Validation Fitness ({final_mean_val:.5f} ± {final_std_val:.5f})', color='#ff7f0e', linewidth=3.0)
        agg_ax.fill_between(iterations, mean_val - std_val, mean_val + std_val, color='#ff7f0e', alpha=0.2)

        if valid_scores:
            agg_ax.axhline(y=mean_test, color='#2ca02c', linestyle='--', linewidth=3.0,
                           label=f'Test Result ({mean_test:.5f} ± {std_test:.5f})')
            agg_ax.fill_between(iterations, mean_test - std_test, mean_test + std_test, color='#2ca02c', alpha=0.1)

        agg_ax.set_title(f'{experiment_name} | {total_folds} Aggregate', pad=15, fontsize=16, fontweight='bold')
        agg_ax.set_xlabel('Iterations', fontweight='bold')
        agg_ax.set_ylabel(f'Fitness ({fitness_metric})', fontweight='bold')
        agg_ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        agg_ax.grid(axis='y', linestyle='--', alpha=0.4)
        agg_ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray', fontsize=11)

        plt.tight_layout()

        if is_final_record:
            cls._save_figure(fig, "CV_Aggregate", experiment_name, fitness_metric)
        plt.show()

    @classmethod
    def plot_rigorous_convergence(cls,
                                  df_trace_history: pd.DataFrame,
                                  experiment_name: str,
                                  is_final_record: bool = False):

        metric_name = df_trace_history['Trace_Metric'].iloc[0] \
            if 'Trace_Metric' in df_trace_history.columns else GlobalSetting.evaluation_function

        df_seed_agg = df_trace_history.groupby(['Fold_ID', 'Iteration']).agg(
            Train_Fit_Mean_by_Seed=('Train_Fitness', 'mean'),
            Val_Fit_Mean_by_Seed=('Val_Fitness', 'mean')
        ).reset_index()


        df_global_eval = df_seed_agg.groupby('Iteration').agg(
            Global_Train_Mean=('Train_Fit_Mean_by_Seed', 'mean'),
            Global_Train_Std=('Train_Fit_Mean_by_Seed', 'std'),
            Global_Val_Mean=('Val_Fit_Mean_by_Seed', 'mean'),
            Global_Val_Std=('Val_Fit_Mean_by_Seed', 'std')
        ).reset_index()

        iters = df_global_eval['Iteration']

        final_train_mean = df_global_eval['Global_Train_Mean'].iloc[-1]
        final_train_std = df_global_eval['Global_Train_Std'].iloc[-1]
        final_val_mean = df_global_eval['Global_Val_Mean'].iloc[-1]
        final_val_std = df_global_eval['Global_Val_Std'].iloc[-1]

        with cls._style_context():
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

            ax.plot(iters, df_global_eval['Global_Train_Mean'],
                    label=f'Training Trace ({final_train_mean:.5f} ± {final_train_std:.5f})',
                    color='#1f77b4', linewidth=2.5)
            ax.fill_between(iters,
                            df_global_eval['Global_Train_Mean'] - df_global_eval['Global_Train_Std'],
                            df_global_eval['Global_Train_Mean'] + df_global_eval['Global_Train_Std'],
                            color='#1f77b4', alpha=0.15, linewidth=0)

            ax.plot(iters, df_global_eval['Global_Val_Mean'],
                    label=f'Validation Trace ({final_val_mean:.5f} ± {final_val_std:.5f})',
                    color='#ff7f0e', linewidth=2.5)
            ax.fill_between(iters,
                            df_global_eval['Global_Val_Mean'] - df_global_eval['Global_Val_Std'],
                            df_global_eval['Global_Val_Mean'] + df_global_eval['Global_Val_Std'],
                            color='#ff7f0e', alpha=0.15, linewidth=0)

            ax.set_title(f'{experiment_name}', pad=15)
            ax.set_xlabel('Iterations')
            ax.set_ylabel(f'Evaluation Metric: {metric_name}')

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray', fontsize=11)

            plt.tight_layout()

            if is_final_record:
                cls._save_figure(fig, "Tuning Tracing/", experiment_name, metric_name)

            plt.show()