import os
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter

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

    @classmethod
    def plot_combined_dynamics(cls,
                               df_precomputed: pd.DataFrame,
                               conv_y_lim: tuple = (0.3, 0.8),
                               scout_y_lim: tuple = (0, 5),
                               is_final_record: bool = False):

        experiment_name = df_precomputed['expr_name'].iloc[0]
        metric_name = df_precomputed['metric_name'].iloc[0]
        iters = df_precomputed['Iteration']

        # === 数据列名提取 ===
        fold_train_lcb_mean = f'train_{metric_name}_LCB_Mean'
        fold_train_lcb_std = f'train_{metric_name}_LCB_std'
        fold_train_lcb_sem = f'train_{metric_name}_LCB_sem'
        seed_train_lcb = f'train_{metric_name}_trace_floor'

        fold_val_lcb_mean = f'val_{metric_name}_LCB_Mean'
        fold_val_lcb_std = f'val_{metric_name}_LCB_std'
        fold_val_lcb_sem = f'val_{metric_name}_LCB_sem'
        seed_val_lcb = f'val_{metric_name}_trace_floor'

        scout_avg = df_precomputed['scout_avg']

        with cls._style_context():
            # 创建 2行1列 的画布，共享X轴，设置高度比例为 2:1
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), dpi=150, sharex=True,
                                           gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.1})

            # ==========================================
            # Subplot 1: 算法收敛动态 (Upper Plot)
            # ==========================================
            # Train 数据流 (精简了Label名称)
            ax1.fill_between(iters,
                             df_precomputed[fold_train_lcb_mean] - df_precomputed[fold_train_lcb_std],
                             df_precomputed[fold_train_lcb_mean] + df_precomputed[fold_train_lcb_std],
                             color='#1f77b4', alpha=0.08, linewidth=0, zorder=1, label='Train STD')

            ax1.fill_between(iters,
                             df_precomputed[fold_train_lcb_mean] - df_precomputed[fold_train_lcb_sem],
                             df_precomputed[fold_train_lcb_mean] + df_precomputed[fold_train_lcb_sem],
                             color='#1f77b4', alpha=0.25, linewidth=1, linestyle='--', edgecolor='#1f77b4', zorder=2,
                             label='Train SEM')

            ax1.plot(iters, df_precomputed[fold_train_lcb_mean],
                     label='Train Mean', color='#1f77b4', linewidth=2.5, zorder=3)

            ax1.plot(iters, df_precomputed[seed_train_lcb],
                     label='Train Lower Bound', color='#1f77b4', linewidth=1.5, linestyle=':', zorder=4)

            # Validation 数据流 (精简了Label名称)
            ax1.fill_between(iters,
                             df_precomputed[fold_val_lcb_mean] - df_precomputed[fold_val_lcb_std],
                             df_precomputed[fold_val_lcb_mean] + df_precomputed[fold_val_lcb_std],
                             color='#ff7f0e', alpha=0.08, linewidth=0, zorder=1, label='Val STD')

            ax1.fill_between(iters,
                             df_precomputed[fold_val_lcb_mean] - df_precomputed[fold_val_lcb_sem],
                             df_precomputed[fold_val_lcb_mean] + df_precomputed[fold_val_lcb_sem],
                             color='#ff7f0e', alpha=0.25, linewidth=1, linestyle='--', edgecolor='#ff7f0e', zorder=2,
                             label='Val SEM')

            ax1.plot(iters, df_precomputed[fold_val_lcb_mean],
                     label='Val Mean', color='#ff7f0e', linewidth=2.5, zorder=3)

            ax1.plot(iters, df_precomputed[seed_val_lcb],
                     label='Val Lower Bound', color='#ff7f0e', linewidth=1.5, linestyle=':', zorder=4)

            # 格式化上图
            cls._format_standard_axes(ax1, title=f'{experiment_name}', xlabel='', ylabel=f'Metric: {metric_name}')

            # 强制应用传入的Y轴上下限
            if conv_y_lim is not None:
                ax1.set_ylim(conv_y_lim)

            # 简化版内嵌图例
            ax1.legend(loc='lower right', ncol=2, framealpha=0.9, edgecolor='gray', fontsize=9, labelspacing=0.4,
                       columnspacing=1.0)

            # ==========================================
            # Subplot 2: 侦查蜂动态 (Lower Plot)
            # ==========================================
            ax2.fill_between(iters, scout_avg, color='#7f8c8d', alpha=0.3)
            ax2.plot(iters, scout_avg, color='#2c3e50', linewidth=1.5, label='Avg Scout Triggers')

            # 格式化下图
            cls._format_standard_axes(ax2, title='', xlabel='Iterations', ylabel='Scout Triggers')

            # 强制应用传入的Y轴上下限
            if scout_y_lim is not None:
                ax2.set_ylim(scout_y_lim)

            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=9)

            # ==========================================
            # 全局 X 轴刻度对齐与导出
            # ==========================================
            max_iter = iters.max()
            ax2.set_xlim(1, max_iter)

            current_ticks = ax2.get_xticks().tolist()
            valid_ticks = [t for t in current_ticks if 1 <= t < max_iter]
            valid_ticks.append(max_iter)
            ax2.set_xticks(valid_ticks)

            plt.tight_layout()

            if is_final_record:
                cls._save_figure(
                    fig=fig,
                    prefix="Combined_Dynamics",
                    experiment_name=experiment_name,
                    fitness_metric=f"{metric_name}_DualPanel"
                )

            plt.show()

    @classmethod
    def plot_2x2_experiment_grid(cls,
                                 dfs: list[pd.DataFrame],
                                 conv_y_lim: tuple = (0, 1),
                                 scout_y_lim: tuple = (0, 10),
                                 global_title: str = "Algorithmic Performance Comparison across Configurations",
                                 is_final_record: bool = False,
                                 expr_name: str = "Trace Result"):
        """
        Takes exactly 4 DataFrames and plots them in a 2x2 grid using Nested GridSpec.
        """
        if len(dfs) != 4:
            raise ValueError("This function requires exactly 4 DataFrames to construct the 2x2 grid.")

        metric_name = dfs[0]['metric_name'].iloc[0]

        with cls._style_context():
            # [1] 创建主画布
            fig = plt.figure(figsize=(16, 12), dpi=250)

            # [2] 定义外部 2x2 网格 (控制四个实验之间的间距)
            outer_grid = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.35)

            global_handles, global_labels = [], []

            left_axes = []

            for idx, df in enumerate(dfs):
                experiment_name = df['expr_name'].iloc[0]
                iters = df['Iteration']

                # 列名提取
                fold_train_lcb_mean, fold_train_lcb_std, fold_train_lcb_sem, seed_train_lcb = \
                    f'train_{metric_name}_LCB_Mean', f'train_{metric_name}_LCB_std', f'train_{metric_name}_LCB_sem', f'train_{metric_name}_trace_floor'
                fold_val_lcb_mean, fold_val_lcb_std, fold_val_lcb_sem, seed_val_lcb = \
                    f'val_{metric_name}_LCB_Mean', f'val_{metric_name}_LCB_std', f'val_{metric_name}_LCB_sem', f'val_{metric_name}_trace_floor'
                scout_avg = df['scout_avg']

                # [3] 定义内部 2x1 网格 (控制单个实验内部 Metric 和 Scout 的间距)
                inner_grid = outer_grid[idx].subgridspec(2, 1, height_ratios=[2.5, 1], hspace=0.08)
                ax_metric = fig.add_subplot(inner_grid[0, 0])
                ax_scout = fig.add_subplot(inner_grid[1, 0], sharex=ax_metric)  # 共享X轴

                # ==========================================
                # 绘制 Upper Plot: Metric
                # ==========================================
                ax_metric.fill_between(iters, df[fold_train_lcb_mean] - df[fold_train_lcb_std],
                                       df[fold_train_lcb_mean] + df[fold_train_lcb_std], color='#1f77b4', alpha=0.08,
                                       linewidth=0, zorder=1, label='Train STD')
                ax_metric.fill_between(iters, df[fold_train_lcb_mean] - df[fold_train_lcb_sem],
                                       df[fold_train_lcb_mean] + df[fold_train_lcb_sem], color='#1f77b4', alpha=0.25,
                                       linewidth=1, linestyle='--', edgecolor='#1f77b4', zorder=2, label='Train SEM')
                ax_metric.plot(iters, df[fold_train_lcb_mean], label='Train Mean', color='#1f77b4', linewidth=2.5,
                               zorder=3)
                ax_metric.plot(iters, df[seed_train_lcb], label='Train Lower Bound', color='#1f77b4', linewidth=1.5,
                               linestyle=':', zorder=4)

                ax_metric.fill_between(iters, df[fold_val_lcb_mean] - df[fold_val_lcb_std],
                                       df[fold_val_lcb_mean] + df[fold_val_lcb_std], color='#ff7f0e', alpha=0.08,
                                       linewidth=0, zorder=1, label='Val STD')
                ax_metric.fill_between(iters, df[fold_val_lcb_mean] - df[fold_val_lcb_sem],
                                       df[fold_val_lcb_mean] + df[fold_val_lcb_sem], color='#ff7f0e', alpha=0.25,
                                       linewidth=1, linestyle='--', edgecolor='#ff7f0e', zorder=2, label='Val SEM')
                ax_metric.plot(iters, df[fold_val_lcb_mean], label='Val Mean', color='#ff7f0e', linewidth=2.5, zorder=3)
                ax_metric.plot(iters, df[seed_val_lcb], label='Val Lower Bound', color='#ff7f0e', linewidth=1.5,
                               linestyle=':', zorder=4)

                # ==========================================
                # 绘制 Lower Plot: Scout
                # ==========================================
                ax_scout.fill_between(iters, scout_avg, color='#7f8c8d', alpha=0.3)
                ax_scout.plot(iters, scout_avg, color='#2c3e50', linewidth=1.5, label='Avg Scout Triggers')

                # ==========================================
                # 坐标轴与标签清理逻辑 (Data-to-Ink 优化)
                # ==========================================
                ax_metric.set_ylim(conv_y_lim)
                ax_scout.set_ylim(scout_y_lim)
                ax_scout.yaxis.set_major_locator(MaxNLocator(integer=True))

                # 学术命名规范：(a), (b), (c), (d)
                panel_label = chr(97 + idx)
                ax_metric.set_title(f"({panel_label}) {experiment_name}", loc='left', fontweight='bold', fontsize=11)

                # 隐藏 Metric 图的 X 轴 Label 和 Ticks
                ax_metric.tick_params(labelbottom=False)
                ax_metric.set_xlabel('')

                # 只有底部的两张图保留 X 轴标题
                if idx >= 2:
                    ax_scout.set_xlabel('Iterations', fontweight='bold')
                else:
                    ax_scout.set_xlabel('')

                # 只有左侧的两张图保留 Y 轴标题
                if idx % 2 == 0:
                    ax_metric.set_ylabel(f'Metric: {metric_name}', fontweight='bold')
                    ax_scout.set_ylabel('Scout Triggers', fontweight='bold')
                    left_axes.extend([ax_metric, ax_scout])
                else:
                    ax_metric.set_ylabel('')
                    ax_scout.set_ylabel('')

                max_iter = iters.max()
                ax_scout.set_xlim(1, max_iter)

                # [4] 仅从第一组实验中提取图例句柄
                if idx == 0:
                    h_metric, l_metric = ax_metric.get_legend_handles_labels()
                    h_scout, l_scout = ax_scout.get_legend_handles_labels()
                    global_handles = h_metric + h_scout
                    global_labels = l_metric + l_scout

            if left_axes:
                fig.align_ylabels(left_axes)

            # [5] 绘制全局统一图例
            # bbox_to_anchor=(0.5, 0.02) 将其锚定在整张画布的正下方中心
            fig.legend(global_handles, global_labels,
                       loc='lower center',
                       bbox_to_anchor=(0.5, 0.02),
                       ncol=5,  # 9个指标分为两行 (5 + 4)
                       frameon=False,  # 移除边框
                       fontsize=10,
                       columnspacing=1.5)

            fig.suptitle(global_title, fontsize=16, fontweight='bold', y=0.95)

            # 调整画布底部边距，为全局图例留出充足物理空间，避免重叠
            fig.subplots_adjust(bottom=0.12, top=0.90)

            if is_final_record:
                cls._save_figure(
                    fig=fig,
                    prefix="Report Figure",
                    experiment_name=expr_name,
                    fitness_metric=f"{metric_name}_GridPanel"
                )

            plt.show()
    @classmethod
    def plot_3x2_experiment_grid(cls,
                                 dfs: list[pd.DataFrame],
                                 conv_y_lim: tuple = (0, 1),
                                 scout_y_lim: tuple = (0, 10),
                                 global_title: str = "Ablation Study on Solution Size and Max Iterations",
                                 is_final_record: bool = False,
                                 expr_name: str = "Trace Result"):
        """
        Takes exactly 6 DataFrames and plots them in a 3x2 grid using Nested GridSpec.
        """
        if len(dfs) != 6:
            raise ValueError(f"This function requires exactly 6 DataFrames, but got {len(dfs)}.")

        metric_name = dfs[0]['metric_name'].iloc[0]

        with cls._style_context():
            # [1] 创建主画布 (高度增加到18，以容纳3行)
            fig = plt.figure(figsize=(16, 18), dpi=150)

            # [2] 定义外部 3x2 网格 (3行，2列)
            outer_grid = fig.add_gridspec(3, 2, wspace=0.15, hspace=0.35)

            global_handles, global_labels = [], []
            left_axes = []  # 收集最左侧的坐标轴用于对齐

            for idx, df in enumerate(dfs):
                experiment_name = df['expr_name'].iloc[0]
                iters = df['Iteration']

                # 列名提取
                fold_train_lcb_mean, fold_train_lcb_std, fold_train_lcb_sem, seed_train_lcb = \
                    f'train_{metric_name}_LCB_Mean', f'train_{metric_name}_LCB_std', f'train_{metric_name}_LCB_sem', f'train_{metric_name}_trace_floor'
                fold_val_lcb_mean, fold_val_lcb_std, fold_val_lcb_sem, seed_val_lcb = \
                    f'val_{metric_name}_LCB_Mean', f'val_{metric_name}_LCB_std', f'val_{metric_name}_LCB_sem', f'val_{metric_name}_trace_floor'
                scout_avg = df['scout_avg']

                # [3] 计算二维网格系的行与列索引
                row = idx // 2
                col = idx % 2

                # [4] 定义内部 2x1 网格 (Metric vs Scout)
                inner_grid = outer_grid[row, col].subgridspec(2, 1, height_ratios=[2.5, 1], hspace=0.08)
                ax_metric = fig.add_subplot(inner_grid[0, 0])
                ax_scout = fig.add_subplot(inner_grid[1, 0], sharex=ax_metric)  # 共享X轴

                # ==========================================
                # 绘制 Upper Plot: Metric
                # ==========================================
                ax_metric.fill_between(iters, df[fold_train_lcb_mean] - df[fold_train_lcb_std],
                                       df[fold_train_lcb_mean] + df[fold_train_lcb_std], color='#1f77b4', alpha=0.08,
                                       linewidth=0, zorder=1, label='Train STD')
                ax_metric.fill_between(iters, df[fold_train_lcb_mean] - df[fold_train_lcb_sem],
                                       df[fold_train_lcb_mean] + df[fold_train_lcb_sem], color='#1f77b4', alpha=0.25,
                                       linewidth=1, linestyle='--', edgecolor='#1f77b4', zorder=2, label='Train SEM')
                ax_metric.plot(iters, df[fold_train_lcb_mean], label='Train Mean', color='#1f77b4', linewidth=2.5,
                               zorder=3)
                ax_metric.plot(iters, df[seed_train_lcb], label='Train Lower Bound', color='#1f77b4', linewidth=1.5,
                               linestyle=':', zorder=4)

                ax_metric.fill_between(iters, df[fold_val_lcb_mean] - df[fold_val_lcb_std],
                                       df[fold_val_lcb_mean] + df[fold_val_lcb_std], color='#ff7f0e', alpha=0.08,
                                       linewidth=0, zorder=1, label='Val STD')
                ax_metric.fill_between(iters, df[fold_val_lcb_mean] - df[fold_val_lcb_sem],
                                       df[fold_val_lcb_mean] + df[fold_val_lcb_sem], color='#ff7f0e', alpha=0.25,
                                       linewidth=1, linestyle='--', edgecolor='#ff7f0e', zorder=2, label='Val SEM')
                ax_metric.plot(iters, df[fold_val_lcb_mean], label='Val Mean', color='#ff7f0e', linewidth=2.5, zorder=3)
                ax_metric.plot(iters, df[seed_val_lcb], label='Val Lower Bound', color='#ff7f0e', linewidth=1.5,
                               linestyle=':', zorder=4)

                # ==========================================
                # 绘制 Lower Plot: Scout
                # ==========================================
                ax_scout.fill_between(iters, scout_avg, color='#7f8c8d', alpha=0.3)
                ax_scout.plot(iters, scout_avg, color='#2c3e50', linewidth=1.5, label='Avg Scout Triggers')

                # ==========================================
                # 坐标轴与标签清理逻辑
                # ==========================================
                ax_metric.set_ylim(conv_y_lim)
                ax_scout.set_ylim(scout_y_lim)
                ax_scout.yaxis.set_major_locator(MaxNLocator(integer=True))

                # 学术命名规范：(a) 到 (f)
                panel_label = chr(97 + idx)

                # 缩短标题，突出核心变量（你可以根据需要修改这行正则表达式或切片逻辑）
                clean_title = experiment_name.replace("cleaned_", "").replace("_Trace", "")
                ax_metric.set_title(f"({panel_label}) {clean_title}", loc='left', fontweight='bold', fontsize=11)

                # 隐藏所有 Metric 图的 X 轴 Label 和 Ticks
                ax_metric.tick_params(labelbottom=False)
                ax_metric.set_xlabel('')

                # 只有底部的两张图（第3行，row == 2）保留 X 轴标题
                if row == 2:
                    ax_scout.set_xlabel('Iterations', fontweight='bold')
                else:
                    ax_scout.set_xlabel('')

                # 只有左侧的三张图（第1列，col == 0）保留 Y 轴标题
                if col == 0:
                    ax_metric.set_ylabel(f'Metric: {metric_name}', fontweight='bold')
                    ax_scout.set_ylabel('Scout Triggers', fontweight='bold')
                    left_axes.extend([ax_metric, ax_scout])
                else:
                    ax_metric.set_ylabel('')
                    ax_scout.set_ylabel('')

                max_iter = iters.max()
                ax_scout.set_xlim(1, max_iter)

                # 仅提取图(a)的句柄用于全局图例
                if idx == 0:
                    h_metric, l_metric = ax_metric.get_legend_handles_labels()
                    h_scout, l_scout = ax_scout.get_legend_handles_labels()
                    global_handles = h_metric + h_scout
                    global_labels = l_metric + l_scout

            # [5] 强制对齐左侧所有的 Y 轴标签（强迫症福音）
            if left_axes:
                fig.align_ylabels(left_axes)

            # [6] 绘制全局统一图例
            fig.legend(global_handles, global_labels,
                       loc='lower center',
                       bbox_to_anchor=(0.5, 0.015),  # 微调以适应18的高度
                       ncol=5,
                       frameon=False,
                       fontsize=11,
                       columnspacing=1.5)

            fig.suptitle(global_title, fontsize=18, fontweight='bold', y=0.96)

            # 调整物理留白：高度从12变成18，对应的百分比也要缩小
            fig.subplots_adjust(bottom=0.07, top=0.93)

            if is_final_record:
                cls._save_figure(
                    fig=fig,
                    prefix="Report Figure",
                    experiment_name= expr_name,
                    fitness_metric=f"{metric_name}_Panel"
                )

            plt.show()
    @classmethod
    def plot_test_results(cls,
                          df_s_all, p_vals_s,
                          df_f_all, df_f_pivot, p_vals_f,
                          target_order,
                          main_model_name   = "ABC RELM CV",
                          metric_name       = 'MCC',
                          show_macro        = True,
                          is_final_record   = False,
                          global_title      = None,
                          deterministic_models = ("LRC", "SVC")):

        with cls._style_context():

            champ_idx = target_order.index(main_model_name)

            n_axes = 2 if show_macro else 1
            h_ratios = [1.5, 1] if n_axes == 2 else [1]
            fig, axes = plt.subplots(n_axes, 1, figsize=(15, 6 * n_axes), dpi=450, sharex=True,
                                     gridspec_kw={'height_ratios': h_ratios})
            if n_axes == 1: axes = [axes]

            def draw_staff_brackets(ax, df_all, p_values, star_color,skip_models=None):
                if skip_models is None:
                    skip_models = []

                valid_df = df_all[~df_all['Model'].isin(skip_models)] if 'Model' in df_all.columns else df_all
                y_max = valid_df[metric_name].max()
                y_min = valid_df[metric_name].min()

                if pd.isna(y_max) or pd.isna(y_min):
                    y_max, y_min = 1.0, 0.0

                y_range = y_max - y_min if y_max != y_min else 0.1

                base_h = y_max + y_range * 0.08
                step_h = y_range * 0.045
                tick_h = y_range * 0.012
                highest_h = base_h

                for i, m in enumerate(target_order):
                    if m == main_model_name or m in skip_models:
                        continue
                    if m not in p_values:
                        continue
                    p_adj, diff_val = p_values[m]

                    if p_adj < 0.05:
                        target_x = i if diff_val > 0 else champ_idx
                        text_color = star_color
                        line_color = 'gray'

                        star_marks = '***' if p_adj < 0.001 else ('**' if p_adj < 0.01 else '*')
                        sign = '+' if diff_val > 0 else '−'
                        display_text = rf"$\mathbf{{({sign}){star_marks}}}$"

                        h = base_h + i * step_h
                        highest_h = max(highest_h, h)

                        ax.plot([i, i, champ_idx, champ_idx], [h - tick_h, h, h, h - tick_h],
                                lw=0.8, c=line_color, alpha=0.4)

                        ax.text(target_x, h + tick_h * 1.5, display_text,
                                ha='center', va='center',
                                color=text_color,
                                fontweight='bold', fontsize=12,
                                bbox=dict(facecolor='white', edgecolor='none', pad=0.1, alpha=0.8))

                ax.set_ylim(y_min - 0.05 * y_range, highest_h + 0.15 * y_range)

            if show_macro:
                ax_m = axes[0]

                df_s_plot = df_s_all.copy()
                mask = df_s_plot['Model'].isin(deterministic_models)
                df_s_plot.loc[mask, metric_name] = np.nan

                sns.boxplot(x='Model', y=metric_name, data=df_s_plot, order=target_order, width=0.35,
                            showmeans=True,
                            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black",
                                       "markersize": 5}, ax=ax_m)
                sns.stripplot(x='Model', y=metric_name, data=df_s_plot, order=target_order, color=".3", size=2.5,
                              alpha=0.2, jitter=True, ax=ax_m)

                draw_staff_brackets(ax_m, df_s_plot, p_vals_s, star_color='darkorange',
                                    skip_models=deterministic_models)

                valid_data = df_s_plot[metric_name].dropna()
                text_y_pos = valid_data.median() if not valid_data.empty else 0.5

                for i, m in enumerate(target_order):
                    if m in deterministic_models:
                        ax_m.text(i, text_y_pos, "N/A\n(Deterministic)", ha='center', va='center',
                                  color='dimgray', style='italic', fontsize=10, fontweight='bold',
                                  bbox=dict(facecolor='whitesmoke', edgecolor='lightgray', boxstyle='round,pad=0.5',
                                            alpha=0.8))

                cls._format_standard_axes(ax_m, ylabel=f"{metric_name} (LCB)")
                ax_m.set_title("  (a) Cross Seeds Stability Analysis", loc='left',
                               fontstyle='italic', fontweight='normal', fontsize=13, color='#444444')
                ax_m.set_xlabel("")
                ax_m.tick_params(labelbottom=False)
                sns.despine(ax=ax_m, left=True)


            ax_e = axes[-1]

            x_coords = np.arange(len(target_order))
            colors = sns.color_palette("husl", n_colors=len(df_f_pivot))
            for i, (fid, row) in enumerate(df_f_pivot.iterrows()):
                ax_e.plot(x_coords, row.values, color=colors[i], alpha=0.35, linewidth=2.5, marker='o', markersize=4,
                          label=f'Fold {fid}')

            g_mean = df_f_pivot.mean(axis=0).values
            ax_e.plot(x_coords, g_mean, color='black', linewidth=3.5, marker='D', markersize=6.5, label='Grand Mean',
                      zorder=10)

            draw_staff_brackets(ax_e, df_f_all, p_vals_f, star_color='royalblue')

            cls._format_standard_axes(ax_e, ylabel=f"Mean {metric_name}")
            ax_e.set_title("  (b) Cross Folds Generalization Analysis", loc='left',
                           fontstyle='italic', fontweight='normal', fontsize=13, color='#444444')
            if ax_e.get_legend():
                ax_e.get_legend().remove()
            ax_e.set_xticks(x_coords)
            ax_e.set_xticklabels(target_order, rotation=15, fontweight='bold')
            sns.despine(ax=ax_e, left=True)

            sig_text = (
                "Significance level:  * p < 0.05   |   ** p < 0.01   |   *** p < 0.001\n"
                f"Effect Direction:    + ({main_model_name} > Baseline)   |   - ({main_model_name} < Baseline)"
            )

            bottom_margin = 0.16 if n_axes == 1 else 0.09

            base_y = 0.02
            line_height = 0.035 if n_axes == 1 else 0.018

            fig.text(0.05, base_y, sig_text,
                     color='dimgray', fontsize=9, style='italic', va='bottom', ha='left')

            fig.text(0.05, base_y + line_height * 1.5, "Paired t-test",
                     color='royalblue', fontsize=10, fontweight='bold', va='bottom', ha='left')

            test_type_y = base_y + line_height * 2.2
            if show_macro:
                fig.text(0.05, test_type_y,
                         f"Wilcoxon signed-rank test: Seed-wise Fold LCB (Mean - {GlobalSetting.cv_punish_coe}*STD))",
                         color='darkorange', fontsize=10, fontweight='bold', va='bottom', ha='left')
            else:
                fig.text(0.05, test_type_y, "Wilcoxon signed-rank test Unapplicable",
                         color='darkorange', fontsize=10, fontweight='bold', va='bottom', ha='left')

            legend_y = bottom_margin * 0.3
            handles, labels = ax_e.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, legend_y),
                       ncol=3, frameon=True, shadow=True, fontsize=9)

            if global_title:
                fig.suptitle(global_title, fontsize=18, fontweight='bold', y=0.95)

            plt.tight_layout(rect=[0, bottom_margin, 1, 0.95])
            fig.align_ylabels(axes)

            if is_final_record:
                cls._save_figure(
                    fig=fig,
                    prefix="Report Figure",
                    experiment_name=global_title,
                    fitness_metric=f"{metric_name}_Panel")
            plt.show()

    @classmethod
    def plot_ABC_algo_tracing(cls,
                              df: pd.DataFrame,
                              title: str = "ABC Algorithms Convergence Results : Train and Validation Floor",
                              is_final_record: bool = False):

        plot_df = df.copy()
        plot_df['Strategy'] = plot_df.apply(
            lambda row: (
                f"Employed bee: {'Algo 3' if row['Employed_Algo3'] else 'Algo 2'}\n"
                f"Onlooker bee: {'Algo 3' if row['Onlooker_Algo3'] else 'Algo 2'}"
            ), axis=1)

        algo_order = [
            "Employed bee: Algo 2\nOnlooker bee: Algo 2",
            "Employed bee: Algo 2\nOnlooker bee: Algo 3",
            "Employed bee: Algo 3\nOnlooker bee: Algo 2",
            "Employed bee: Algo 3\nOnlooker bee: Algo 3"
        ]

        with cls._style_context():
            fig, (ax_train, ax_val) = plt.subplots(nrows=2, ncols=1, figsize=(12, 9.6), dpi=450, sharex=True)

            sns.boxplot(data=plot_df, x='Strategy', y='train_MCC_trace_floor', ax=ax_train,
                        color='#1f77b4', boxprops=dict(alpha=0.5), order=algo_order)
            sns.swarmplot(data=plot_df, x='Strategy', y='train_MCC_trace_floor', ax=ax_train,
                          color=".25", size=5, alpha=0.7, order=algo_order)

            cls._format_standard_axes(ax_train, xlabel='', ylabel='MCC Floor (Train)')
            ax_train.set_title("  (a) Training Convergence", loc='left',
                               fontstyle='italic', fontweight='normal', fontsize=13, color='#444444')
            ax_train.set_xlim(-0.5, len(algo_order) - 0.5)
            sns.despine(ax=ax_train, left=False)

            sns.boxplot(data=plot_df, x='Strategy', y='val_MCC_trace_floor', ax=ax_val,
                        color='#ff7f0e', boxprops=dict(alpha=0.5), order=algo_order)
            sns.swarmplot(data=plot_df, x='Strategy', y='val_MCC_trace_floor', ax=ax_val,
                          color=".25", size=5, alpha=0.7, order=algo_order)

            cls._format_standard_axes(ax_val, xlabel='ABC Algorithm Configurations',
                                      ylabel='MCC Floor (Validation)')
            ax_val.set_title("  (b) Validation Convergence", loc='left',
                             fontstyle='italic', fontweight='normal', fontsize=13, color='#444444')
            ax_val.set_xlim(-0.5, len(algo_order) - 0.5)
            ax_val.set_xticks(range(len(algo_order)))
            sns.despine(ax=ax_val, left=False)

            # --- Layout Engine ---
            if title:
                fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.subplots_adjust(hspace=0.15)
            fig.align_ylabels((ax_train, ax_val))

            if is_final_record:
                cls._save_figure(
                    fig=fig,
                    prefix="Tracing Figure",
                    experiment_name=title,
                    fitness_metric="Dual_Subplot"
                )
            plt.show()

    @classmethod
    def plot_ABC_limit_ratio_tracing(cls,
                                     df: pd.DataFrame,
                                     title: str = "ABC Limit Ratio Convergence Results : Train and Validation Floor",
                                     is_final_record: bool = False):

        plot_df = df.copy()
        plot_df['Strategy'] = plot_df['L/SN'].apply(lambda x: f"L/SN Ratio: {x:.1f}")

        sorted_ratios = sorted(plot_df['L/SN'].unique())
        category_order = [f"L/SN Ratio: {r:.1f}" for r in sorted_ratios]

        with cls._style_context():
            fig, (ax_train, ax_val) = plt.subplots(nrows=2, ncols=1, figsize=(12, 9.6), dpi=450, sharex=True)

            sns.boxplot(data=plot_df, x='Strategy', y='train_MCC_trace_floor', ax=ax_train,
                        color='#1f77b4', boxprops=dict(alpha=0.5), order=category_order)
            sns.swarmplot(data=plot_df, x='Strategy', y='train_MCC_trace_floor', ax=ax_train,
                          color=".25", size=5, alpha=0.7, order=category_order)

            cls._format_standard_axes(ax_train, xlabel='', ylabel='MCC Floor (Train)')
            ax_train.set_title("  (a) Training Convergence", loc='left',
                               fontstyle='italic', fontweight='normal', fontsize=13, color='#444444')
            ax_train.set_xlim(-0.5, len(category_order) - 0.5)
            sns.despine(ax=ax_train, left=False)

            sns.boxplot(data=plot_df, x='Strategy', y='val_MCC_trace_floor', ax=ax_val,
                        color='#ff7f0e', boxprops=dict(alpha=0.5), order=category_order)
            sns.swarmplot(data=plot_df, x='Strategy', y='val_MCC_trace_floor', ax=ax_val,
                          color=".25", size=5, alpha=0.7, order=category_order)

            cls._format_standard_axes(ax_val, xlabel='Trial Limit / Solution Size Ratio',
                                      ylabel='MCC Floor (Validation)')
            ax_val.set_title("  (b) Validation Convergence", loc='left',
                             fontstyle='italic', fontweight='normal', fontsize=13, color='#444444')
            ax_val.set_xlim(-0.5, len(category_order) - 0.5)
            ax_val.set_xticks(range(len(category_order)))
            sns.despine(ax=ax_val, left=False)

            # --- Layout Engine ---
            if title:
                fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.subplots_adjust(hspace=0.15)
            fig.align_ylabels((ax_train, ax_val))

            if is_final_record:
                cls._save_figure(
                    fig=fig,
                    prefix="Tracing Figure",
                    experiment_name=title,
                    fitness_metric="Dual_Subplot"
                )
            plt.show()
    @classmethod
    def plot_ablation_diagnostic_panel(cls,
                                       df: pd.DataFrame,
                                       sn_range     : tuple = (10, 250),
                                       mi_range     : tuple = (50, 1000),
                                       gf_sigma     : float = 0.6,
                                       global_title : str   = "Ablation Diagnostic Panel: Empirical vs. Topological Manifold",
                                       is_final_record: bool = False,
                                       expr_name    : str   = "SN_MI_Diagnostic_1x2_Panel"):
        plot_df = df.copy()

        # [1] ROI Constraint
        mask = (plot_df['Solution_Size'] >= sn_range[0]) & (plot_df['Solution_Size'] <= sn_range[1]) & \
               (plot_df['Max_Iteration'] >= mi_range[0]) & (plot_df['Max_Iteration'] <= mi_range[1])
        roi_df = plot_df[mask]

        # [2] Pivot & Math Sorting
        pivot_df = roi_df.pivot_table(index='Solution_Size', columns='Max_Iteration',
                                      values='val_MCC_trace_floor', aggfunc='mean')
        pivot_df = pivot_df.sort_index(ascending=True).sort_index(axis=1, ascending=True)

        # [3] NaN Poisoning Defense & Matrix Calculations
        raw_matrix = pivot_df.to_numpy()
        nan_mask = np.isnan(raw_matrix)

        # 临时用周围的有效值填补 NaN，防止高斯滤波崩溃
        filled_df = pivot_df.ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0)

        # 提取高斯平滑流形
        smoothed_matrix = gaussian_filter(filled_df.to_numpy(), sigma=gf_sigma, mode='nearest')

        # 算完后，把原本是空白的区域重新变回 NaN
        smoothed_matrix[nan_mask] = np.nan

        # [4] DataFrame Reconstruction (Y轴反转)
        raw_df = pd.DataFrame(raw_matrix, index=pivot_df.index, columns=pivot_df.columns).sort_index(ascending=False)
        smoothed_df = pd.DataFrame(smoothed_matrix, index=pivot_df.index, columns=pivot_df.columns).sort_index(
            ascending=False)

        with cls._style_context():
            # [5] 1x3 Grid Architecture: Left (1) : Right (1) : Colorbar (0.04)
            # 引入独立的色阶坐标轴，彻底解决 square=True 导致的大小不一致问题
            fig, axes = plt.subplots(1, 3, figsize=(15, 8), dpi=250,
                                     gridspec_kw={'width_ratios': [1, 1, 0.04], 'wspace': 0.05})

            val_min = raw_df.min().min()
            val_max = raw_df.max().max()

            # ==========================================
            # Panel (a): Raw Empirical (axes[0])
            # ==========================================
            sns.heatmap(raw_df, annot=True, fmt=".4f", cmap="YlOrBr",
                        vmin=val_min, vmax=val_max,
                        linewidths=1.0, linecolor='white', square=True,
                        cbar=False, ax=axes[0])

            axes[0].set_title("(a) Empirical Observation\n[ Metric: Validation MCC Floor ]",
                              loc='left', fontstyle='italic', fontsize=13, color='#444444', pad=15)
            axes[0].set_ylabel("Solution Size (Exploration)", fontweight='bold', fontsize=12)
            axes[0].set_xlabel("Max Iteration (Exploitation)", fontweight='bold', fontsize=12)
            axes[0].tick_params(axis='y', rotation=0)

            # ==========================================
            # Panel (b): Gaussian Smoothed Manifold (axes[1])
            # ==========================================
            sns.heatmap(smoothed_df, annot=True, fmt=".4f", cmap="YlOrBr",
                        vmin=val_min, vmax=val_max,
                        linewidths=1.0, linecolor='white', square=True,
                        yticklabels=False,
                        cbar=True,
                        cbar_ax=axes[2],
                        cbar_kws={'label': 'MCC Score (Unified Scale)'},
                        ax=axes[1])

            axes[1].set_title("(b) Gaussian-Smoothed Manifold\n[ Metric: Expected MCC Floor ]",
                              loc='left', fontstyle='italic', fontsize=13, color='#444444', pad=15)
            axes[1].set_ylabel("")
            axes[1].tick_params(axis='y', left=False, labelleft=False)
            axes[1].set_xlabel("Max Iteration (Exploitation)", fontweight='bold', fontsize=12)

            # [6] Global Typographic Cleanup
            axes[0].tick_params(axis='x', rotation=0)
            axes[1].tick_params(axis='x', rotation=0)

            fig.suptitle(global_title, fontsize=17, fontweight='bold', y=1.05)

            if is_final_record:
                cls._save_figure(
                    fig=fig,
                    prefix="Diagnostic_Panel",
                    experiment_name=expr_name,
                    fitness_metric="1x2_Empirical_vs_Smoothed"
                )
            plt.subplots_adjust(top=0.85, bottom=0.15)
            plt.show()

    @classmethod
    def plot_residual_diagnostic_heatmap(cls,
                                         df: pd.DataFrame,
                                         sn_range: tuple = (10, 250),
                                         mi_range: tuple = (50, 1000),
                                         gf_sigma: float = 0.6,
                                         title: str = "Residual Diagnostic Heatmap: Deviation from Gaussian Manifold",
                                         is_final_record: bool = False,
                                         expr_name: str = "SN_MI_Residual_Analysis"):
        """
        Calculates and plots the residual matrix (Raw - Smoothed) to isolate
        stochastic noise from the underlying deterministic manifold.
        Incorporates NaN-defense for discontinuous grid search spaces.
        """
        plot_df = df.copy()

        # [1] ROI Constraint
        mask = (plot_df['Solution_Size'] >= sn_range[0]) & (plot_df['Solution_Size'] <= sn_range[1]) & \
               (plot_df['Max_Iteration'] >= mi_range[0]) & (plot_df['Max_Iteration'] <= mi_range[1])
        roi_df = plot_df[mask]

        # [2] Pivot & Math Sorting
        pivot_df = roi_df.pivot_table(index='Solution_Size', columns='Max_Iteration',
                                      values='val_MCC_trace_floor', aggfunc='mean')
        pivot_df = pivot_df.sort_index(ascending=True).sort_index(axis=1, ascending=True)

        # [3] NaN Poisoning Defense & Calculate Residuals
        raw_matrix = pivot_df.to_numpy()
        nan_mask = np.isnan(raw_matrix)

        # 应用拓扑填充以计算无损的高斯流形
        filled_df = pivot_df.ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0)
        smoothed_matrix = gaussian_filter(filled_df.to_numpy(), sigma=gf_sigma, mode='nearest')

        # 剥离噪声：原始观测 - 平滑流形
        residual_matrix = raw_matrix - smoothed_matrix

        # 将无实验数据的区域重新设为空白，保证严谨性
        residual_matrix[nan_mask] = np.nan

        residual_df = pd.DataFrame(residual_matrix, index=pivot_df.index, columns=pivot_df.columns)
        residual_df = residual_df.sort_index(ascending=False)

        with cls._style_context():
            fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

            # [4] Rendering: Diverging Colormap centered at zero
            # 核心修复：使用 np.nanmin 和 np.nanmax 绕过 NaN 值，防止色阶 limit 计算崩溃
            limit = max(abs(np.nanmin(residual_matrix)), abs(np.nanmax(residual_matrix)))

            sns.heatmap(residual_df,
                        annot=True, fmt=".4f",
                        cmap="coolwarm", center=0, vmin=-limit, vmax=limit,
                        linewidths=1.0, linecolor='white', square=True,
                        cbar_kws={'label': 'Residual Error (Raw - Smoothed)'},
                        ax=ax)

            # [5] Typographic Engine (统一排版隐喻)
            ax.set_title(title, fontweight='bold', fontsize=16, pad=20, color='#444444')
            ax.set_xlabel("Max Iteration (Exploitation)", fontweight='bold', fontsize=12)
            ax.set_ylabel("Solution Size (Exploration)", fontweight='bold', fontsize=12)

            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            plt.tight_layout()

            if is_final_record:
                cls._save_figure(fig=fig, prefix="Ablation Heatmap",
                                 experiment_name=expr_name, fitness_metric="Residual_Matrix_CoolWarm")
            plt.show()