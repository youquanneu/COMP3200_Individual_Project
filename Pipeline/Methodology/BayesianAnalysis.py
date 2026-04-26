import os
from contextlib import contextmanager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from baycomp import CorrelatedTTest

from Pipeline.Global.GlobalSetting import GlobalSetting


class BayesianAnalysis:
    seaborn_theme = {
        "style": "whitegrid",
        "context": "paper",
        "font_scale": 1.2
    }

    rc_params_standard = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'figure.titlesize': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.axisbelow': True
    }

    @classmethod
    @contextmanager
    def _style_context(cls):
        sns.set_theme(**cls.seaborn_theme)
        with plt.rc_context(cls.rc_params_standard):
            yield

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
        print(f"[I/O Trace] Bayesian Summary Figure exported: {file_path}")

    @classmethod
    def run_bayesian_evaluation(cls,
                                model_dict: dict,
                                champion_model: str,
                                baselines: list,
                                metric_name: str = 'MCC',
                                rope_radius: float = 0.02,
                                cv_folds: int = 5,
                                certainty_threshold: float = 0.95,
                                title:str = None,
                                expr_name: str = "Bayesian_Summary",
                                is_final_record: bool = False,
                                title_on = True):

        champ_df = pd.read_csv(model_dict[champion_model])
        champ_scores = champ_df[metric_name].values

        results_summary = []

        # 1. 批量计算所有模型的概率
        for base_model in baselines:
            base_df = pd.read_csv(model_dict[base_model])
            base_scores = base_df[metric_name].values

            min_len = min(len(champ_scores), len(base_scores))
            x = champ_scores[:min_len]
            y = base_scores[:min_len]
            runs = len(x) // cv_folds

            # 贝叶斯计算 (不直接绘图，只提取概率)
            posterior = CorrelatedTTest(x, y, rope=rope_radius, runs=runs)
            p_champ_wins, p_rope, p_base_wins = posterior.probs()

            decision = 'Win' if p_champ_wins > certainty_threshold else ('Lose' if p_base_wins > certainty_threshold else 'Inconclusive')

            results_summary.append({
                'Baseline': base_model,
                'P_Win': p_champ_wins,
                'P_ROPE': p_rope,
                'P_Lose': p_base_wins,
                'Decision': decision
            })

        with cls._style_context():
            fig_height = max(4.0, len(baselines) * 0.8)
            fig, ax = plt.subplots(figsize=(10, fig_height), dpi=450)

            # 准备数据作图
            y_pos = np.arange(len(baselines))
            models = [res['Baseline'] for res in results_summary]

            p_wins = np.array([res['P_Win'] for res in results_summary]) * 100
            p_ropes = np.array([res['P_ROPE'] for res in results_summary]) * 100
            p_loses = np.array([res['P_Lose'] for res in results_summary]) * 100

            color_win = '#2ca02c'
            color_rope = '#cccccc'
            color_lose = '#d62728'

            bars_win = ax.barh(y_pos, p_wins, color=color_win, edgecolor='white', height=0.6,
                               label=f'{champion_model} Wins')
            bars_rope = ax.barh(y_pos, p_ropes, left=p_wins, color=color_rope, edgecolor='white', height=0.6,
                                label='Practically Equivalent (ROPE)')
            bars_lose = ax.barh(y_pos, p_loses, left=p_wins + p_ropes, color=color_lose, edgecolor='white', height=0.6,
                                label=f'{champion_model} Loses')


            for i, bars in enumerate([bars_win, bars_rope, bars_lose]):
                text_color = 'white' if i == 0 else 'black'
                for bar in bars:
                    width = bar.get_width()
                    if width > 5.0:
                        x_center = bar.get_x() + width / 2
                        y_center = bar.get_y() + bar.get_height() / 2
                        ax.text(x_center, y_center, f'{width:.1f}%', ha='center', va='center',
                                color=text_color, fontweight='bold', fontsize=10)

            threshold_pct = certainty_threshold * 100
            ax.axvline(threshold_pct, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)

            ax.text(threshold_pct+1, len(baselines)-0.2,
                    f'P > {certainty_threshold:.0%}',
                    color='black', fontsize= 8, fontweight='bold',style='italic',
                    ha='center', va='bottom')

            ax.set_yticks(y_pos)
            formatted_models = [m.replace(" (", "\n(") if " (" in m else m for m in models]
            ax.set_yticklabels(formatted_models, fontweight='bold', fontsize=8, rotation=45, ha='right')

            ax.set_xlim(0, 100)
            ax.set_xlabel('Posterior Probability (%)', fontweight='bold')

            if title_on:
                main_title = title
                sub_title = f"Metric: {metric_name} | ROPE: ±{rope_radius} "
                ax.set_title(f"{main_title}\n", fontsize=13, fontweight='bold', pad=10)
                ax.text(0.5, 1.02, sub_title, transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=10, style='italic', color='#555555')

            ax.legend(loc='upper center',
                      bbox_to_anchor=(0.5, -0.08),
                      ncol=3,
                      frameon=False,
                      fontsize=9,
                      handletextpad=0.5,
                      columnspacing=1.0)

            # 去除冗余的边框
            sns.despine(left=True, bottom=True)
            ax.xaxis.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()

            if is_final_record:
                cls._save_figure(fig, "Report Figure", expr_name, f"Bayesian_Summary_{metric_name}")

            plt.show()

        summary_df = pd.DataFrame(results_summary).set_index('Baseline')

        return summary_df