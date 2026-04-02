import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import seaborn as sns
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

        target_dir = GlobalSetting.get_figure_dir()
        safe_filename = (experiment_name
                         .replace(" ", "_")
                         .replace(":", "")
                         .replace("/", "-"))

        sub_dir_path = os.path.join(target_dir, prefix)
        os.makedirs(sub_dir_path, exist_ok=True)

        file_path = os.path.join(sub_dir_path, f"{safe_filename}_{fitness_metric}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[I/O Trace] Figure exported successfully: {file_path}")

    @classmethod
    def plot_rigorous_convergence(cls,
                                  df_precomputed: pd.DataFrame,
                                  is_final_record: bool = False):

        experiment_name = df_precomputed['expr_name'].iloc[0]
        metric_name     = df_precomputed['metric_name'].iloc[0]
        iters           = df_precomputed['Iteration']

        fold_train_lcb_mean = f'train_{metric_name}_LCB_Mean'
        fold_train_lcb_std  = f'train_{metric_name}_LCB_std'
        fold_train_lcb_sem  = f'train_{metric_name}_LCB_sem'
        seed_train_lcb      = f'train_{metric_name}_trace_floor'

        fold_val_lcb_mean   = f'val_{metric_name}_LCB_Mean'
        fold_val_lcb_std    = f'val_{metric_name}_LCB_std'
        fold_val_lcb_sem    = f'val_{metric_name}_LCB_sem'
        seed_val_lcb        = f'val_{metric_name}_trace_floor'

        final_train_mean    = df_precomputed[fold_train_lcb_mean].iloc[-1]
        final_train_std     = df_precomputed[fold_train_lcb_std].iloc[-1]
        final_train_sem     = df_precomputed[fold_train_lcb_sem].iloc[-1]
        final_train_lcb     = df_precomputed[seed_train_lcb].iloc[-1]

        final_val_mean      = df_precomputed[fold_val_lcb_mean].iloc[-1]
        final_val_std       = df_precomputed[fold_val_lcb_std].iloc[-1]
        final_val_sem       = df_precomputed[fold_val_lcb_sem].iloc[-1]
        final_val_lcb       = df_precomputed[seed_val_lcb].iloc[-1]

        with cls._style_context():
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.fill_between(iters,
                            df_precomputed[fold_train_lcb_mean] - df_precomputed[fold_train_lcb_std],
                            df_precomputed[fold_train_lcb_mean] + df_precomputed[fold_train_lcb_std],
                            color='#1f77b4', alpha=0.08, linewidth=0, zorder=1,
                            label=f'Train Inter-Seed_LCB STD (± {final_train_std:.4f})')

            ax.fill_between(iters,
                            df_precomputed[fold_train_lcb_mean] - df_precomputed[fold_train_lcb_sem],
                            df_precomputed[fold_train_lcb_mean] + df_precomputed[fold_train_lcb_sem],
                            color='#1f77b4', alpha=0.25, linewidth=1, linestyle='--', edgecolor='#1f77b4', zorder=2,
                            label=f'Train Mean Confidence SEM (± {final_train_sem:.4f})')

            ax.plot(iters, df_precomputed[fold_train_lcb_mean],
                    label=f'Train Inter-Seed_LCB Mean ({final_train_mean:.4f})',
                    color='#1f77b4', linewidth=2.5, zorder=3)

            ax.plot(iters, df_precomputed[seed_train_lcb],
                    label=f'Train Pessimistic Lower Bound ({final_train_lcb:.4f})',
                    color='#1f77b4', linewidth=1.5, linestyle=':', zorder=4)

            ax.fill_between(iters,
                            df_precomputed[fold_val_lcb_mean] - df_precomputed[fold_val_lcb_std],
                            df_precomputed[fold_val_lcb_mean] + df_precomputed[fold_val_lcb_std],
                            color='#ff7f0e', alpha=0.08, linewidth=0, zorder=1,
                            label=f'Val Inter-Seed_LCB STD (± {final_val_std:.4f})')

            ax.fill_between(iters,
                            df_precomputed[fold_val_lcb_mean] - df_precomputed[fold_val_lcb_sem],
                            df_precomputed[fold_val_lcb_mean] + df_precomputed[fold_val_lcb_sem],
                            color='#ff7f0e', alpha=0.25, linewidth=1, linestyle='--', edgecolor='#ff7f0e', zorder=2,
                            label=f'Val Mean Confidence SEM (± {final_val_sem:.4f})')

            ax.plot(iters, df_precomputed[fold_val_lcb_mean],
                    label=f'Val Inter-Seed_LCB Mean ({final_val_mean:.4f})',
                    color='#ff7f0e', linewidth=2.5, zorder=3)

            ax.plot(iters, df_precomputed[seed_val_lcb],
                    label=f'Val Pessimistic Lower Bound ({final_val_lcb:.4f})',
                    color='#ff7f0e', linewidth=1.5, linestyle=':', zorder=4)

            cls._format_standard_axes(ax,
                                      title=f'{experiment_name}',
                                      xlabel='Iterations',
                                      ylabel=f'Evaluation Metric: {metric_name}')

            max_iter = iters.max()
            ax.set_xlim(1, max_iter)

            current_ticks = ax.get_xticks().tolist()
            valid_ticks = [t for t in current_ticks if 1 <= t < max_iter]
            valid_ticks.append(max_iter)
            ax.set_xticks(valid_ticks)

            y_offset = -0.22

            ax.text(0.0, y_offset - 0.01,
                    'Note: Shaded bands represent\nVariance (±1 STD) & Confidence (±1 SEM).',
                    transform=ax.transAxes,
                    fontsize=8.5,
                    color='#555555',
                    style='italic',
                    ha='left',
                    va='top')

            ax.legend(loc='upper right',
                      bbox_to_anchor=(1.0, y_offset),
                      ncol=2,
                      framealpha=0.9,
                      edgecolor='gray',
                      fontsize=9,
                      labelspacing=0.5,
                      columnspacing=1.0)

            plt.tight_layout()

            fig.subplots_adjust(bottom=0.30)

            if is_final_record:
                cls._save_figure(
                    fig     = fig,
                    prefix  = "Iteration Tracing",
                    experiment_name = experiment_name,
                    fitness_metric  = f"{metric_name}_DualBand"
                )

            plt.show()

    @classmethod
    def plot_scout_dynamics(cls, df_precomputed: pd.DataFrame, is_final_record: bool = False):
        experiment_name = df_precomputed['expr_name'].iloc[0]
        iters = df_precomputed['Iteration']
        scout_avg = df_precomputed['scout_avg']

        with cls._style_context():
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

            ax.fill_between(iters, scout_avg, color='#7f8c8d', alpha=0.3)
            ax.plot(iters, scout_avg, color='#2c3e50', linewidth=1.5, label='Avg Scout Triggers')

            cls._format_standard_axes(ax,
                                      title=f"Scout Bee Dynamics: {experiment_name}",
                                      xlabel='Iterations',
                                      ylabel='Avg Triggers / Seed')

            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            plt.tight_layout()

            if is_final_record:
                cls._save_figure(
                    fig     = fig,
                    prefix  = "Scouts Tracing",
                    experiment_name = experiment_name,
                    fitness_metric  = f"Scout_Triggers"
                )
            plt.show()