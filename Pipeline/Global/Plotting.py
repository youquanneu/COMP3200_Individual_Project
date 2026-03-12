import os

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
    def _apply_seaborn_theme(cls):
        """Internal helper to apply Seaborn base layer."""
        sns.set_theme(**cls.seaborn_theme)

    @classmethod
    def plot_abc_dashboard(cls, convergence_df: pd.DataFrame,
                           scout_df: pd.DataFrame,
                           experiment_name: str,
                           results_df: pd.DataFrame = None,
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
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16),
                                           gridspec_kw={'height_ratios': [6, 1]},
                                           facecolor='white')

            # === NEW: The Global Dashboard Title ===
            fig.suptitle(f'ABC-ELM Telemetry | {experiment_name}', y=0.98, fontsize=18, fontweight='bold')

            iterations = np.arange(1, len(convergence_df) + 1)

            # --- TOP PLOT: Convergence Trend ---
            for seed_col in convergence_df.columns:
                ax1.plot(iterations, convergence_df[seed_col],
                         linestyle=':', linewidth=1, alpha=0.75)

            mean_fitness = convergence_df.mean(axis=1)
            ax1.plot(iterations, mean_fitness,
                     color='#D62728', linewidth=3.5, label='Average Convergence')

            # === Testing Result Channel with Explicit Std Value ===
            if results_df is not None and fitness_metric in results_df.columns:
                test_mean = results_df[fitness_metric].mean()
                test_std = results_df[fitness_metric].std()

                # 1. Plot Mean Line
                ax1.axhline(y=test_mean, color='#2ca02c', linestyle='--', linewidth=2.5,
                            label=f'Final Test Mean ({test_mean:.4f})')

                # 2. Plot shaded region and EXPLICITLY state the standard deviation in the legend
                ax1.axhspan(ymin=test_mean - test_std, ymax=test_mean + test_std,
                            color='#2ca02c', alpha=0.15,
                            label=f'Test Variance ($\pm 1\sigma$, $\sigma={test_std:.4f}$)')

                last_x = iterations[-1]

                # 3. Update annotations to clarify both the bound and the std step
                ax1.text(x=last_x, y=test_mean + test_std,
                         s=f'+1 $\sigma$ bound ({test_mean + test_std:.3f})',
                         color='#1b5e20',
                         va='bottom', ha='right',
                         fontweight='bold', fontsize=11)

                ax1.text(x=last_x, y=test_mean - test_std,
                         s=f'-1 $\sigma$ bound ({test_mean - test_std:.3f})',
                         color='#1b5e20',
                         va='top', ha='right',
                         fontweight='bold', fontsize=11)


            ax1.set_ylim(0.7, 0.9)

            # DOWNGRADED: Now just a subplot label, not the main title
            ax1.set_title('Phase 1: Convergence Trajectory', pad=15)
            ax1.set_ylabel(f'Fitness ({fitness_metric})')

            ax1.grid(axis='y', linestyle='--', alpha=0.4)
            ax1.grid(axis='x', visible=False)

            ax1.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

            # --- BOTTOM PLOT: Scout Deployments ---
            total_scouts = scout_df.sum(axis=1)
            ax2.bar(iterations, total_scouts, color='#1f77b4', alpha=0.8, edgecolor='black')

            # DOWNGRADED: Subplot label
            ax2.set_title('Phase 2: Scout Bee Interventions', pad=15)
            ax2.set_xlabel('Iteration / Bee Cycle')
            ax2.set_ylabel('Total Triggers')

            ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
            ax2.grid(axis='y', linestyle='--', alpha=0.4)

            plt.tight_layout()
            fig.subplots_adjust(top=0.90, hspace=0.35)

            if is_final_record:
                target_dir = GlobalSetting.get_figure_dir()
                safe_filename = experiment_name.replace(" ", "_").replace(":", "").replace("/", "-")
                file_path = os.path.join(target_dir, f"ABC_Telemetry_{safe_filename}_{fitness_metric}.png")

                plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
                print(f"[I/O Trace] Figure exported successfully: {file_path}")

            plt.show()