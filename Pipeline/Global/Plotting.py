import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
    def _apply_seaborn_theme(cls):
        """Internal helper to apply Seaborn base layer."""
        sns.set_theme(**cls.seaborn_theme)

    @classmethod
    def plot_abc_dashboard(cls, convergence_df: pd.DataFrame,
                           scout_df: pd.DataFrame,
                           experiment_name: str,
                           fitness_metric: str = None,
                           is_final_record = False):
        """
        Generates a standardized telemetry dashboard.
        Uses rc_context to prevent global matplotlib state pollution.
        """
        cls._apply_seaborn_theme()

        if fitness_metric is None:
            fitness_metric = GlobalSetting.evaluation_function

        with plt.rc_context(cls.rc_params_standard):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]},
                                           facecolor='white')

            # === NEW: The Global Dashboard Title ===
            fig.suptitle(f'ABC-ELM Telemetry | {experiment_name}', y=0.98, fontsize=18, fontweight='bold')

            iterations = np.arange(1, len(convergence_df) + 1)

            # --- TOP PLOT: Convergence Trend ---
            for seed_col in convergence_df.columns:
                ax1.plot(iterations, convergence_df[seed_col],
                         linestyle=':', linewidth=1.5, alpha=0.7, label=f'Seed {seed_col}')

            mean_fitness = convergence_df.mean(axis=1)
            ax1.plot(iterations, mean_fitness,
                     color='#D62728', linewidth=3.5, label='Average Convergence')

            # DOWNGRADED: Now just a subplot label, not the main title
            ax1.set_title('Phase 1: Convergence Trajectory', pad=15)
            ax1.set_ylabel(f'Fitness ({fitness_metric})')
            ax1.grid(axis='y', linestyle='--', alpha=0.5)
            ax1.grid(axis='x', visible=False)
            ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

            # --- BOTTOM PLOT: Scout Deployments ---
            total_scouts = scout_df.sum(axis=1)
            ax2.bar(iterations, total_scouts, color='#1f77b4', alpha=0.8, edgecolor='black')

            # DOWNGRADED: Subplot label
            ax2.set_title('Phase 2: Scout Bee Interventions', pad=15)
            ax2.set_xlabel('Iteration / Bee Cycle')
            ax2.set_ylabel('Total Triggers')
            ax2.set_xticks(iterations)
            ax2.grid(axis='y', linestyle='--', alpha=0.5)
            ax2.grid(axis='x', visible=False)

            # Adjust layout to make room for the new suptitle
            plt.tight_layout()
            fig.subplots_adjust(top=0.90)

            if is_final_record:
                target_dir = GlobalSetting.get_figure_dir()
                safe_filename = experiment_name.replace(" ", "_").replace(":", "").replace("/", "-")
                file_path = os.path.join(target_dir, f"ABC_Telemetry_{safe_filename}_{fitness_metric}.png")

                plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
                print(f"[I/O Trace] Figure exported successfully: {file_path}")

            plt.show()