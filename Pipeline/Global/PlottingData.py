import math
import textwrap


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy import stats

from Pipeline.Global.Plotting import Plotting


class PlottingData(Plotting):

    @classmethod
    def plot_anthropometric_distribution(
            cls,
            df: pd.DataFrame,
            title: str = "Anthropometric Distribution",
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Plots the demographic/anthropometric distribution characteristics.
        (Migrated from DataAnalysis.ipynb)
        """
        # 1. Defensive copy to prevent mutating the original DataFrame
        df_viz = df.copy()

        # Rename columns for better visualization labels
        rename_map = {
            'Height': 'Height (cm)',
            'Weight': 'Weight (kg)'
        }
        df_viz = df_viz.rename(columns={k: v for k, v in rename_map.items() if k in df_viz.columns})

        # Handle gender labels if the column exists
        if 'Gender' in df_viz.columns:
            df_viz['Gender_Label'] = df_viz['Gender'].map({0: 'Male', 1: 'Female'})

        # 2. Invoke the unified styling context from the parent class (Plotting)
        with cls._style_context():

            # 3. Rendering engine
            g = sns.pairplot(
                df_viz,
                hue='Gender_Label' if 'Gender_Label' in df_viz.columns else None,
                vars=[col for col in ['Age', 'Height (cm)', 'Weight (kg)'] if col in df_viz.columns],
                kind='reg',
                diag_kind='kde',
                palette={'Male': 'tab:blue', 'Female': 'tab:red'} if 'Gender_Label' in df_viz.columns else None,
                plot_kws={'scatter_kws': {'alpha': 0.4, 's': 15, 'edgecolor': 'none'}, 'line_kws': {'linewidth': 2}},
                diag_kws={'fill': True, 'alpha': 0.3},
                height=2.5,
                aspect=1.6,
                corner=True
            )

            g.figure.set_dpi(450)

            # --- NEW: Academic Legend Formatting ---
            if 'Gender_Label' in df_viz.columns:
                sns.move_legend(
                    g,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.05),  # Positioned at the bottom center of the figure
                    ncol=2,  # Split into 2 columns (Horizontal)
                    title=None,  # Remove 'Gender_Label' title
                    frameon=False  # Academic standard: no distracting borders
                )
                # Adjust bottom margin to prevent the legend from being cut off
                g.figure.subplots_adjust(bottom=0.15)

            # 4. Title control logic
            if title_on:
                g.figure.suptitle(title, y=1.02, fontsize=16, fontweight='bold')

            if is_final_record:
                cls._save_figure(
                    fig=g.figure,
                    prefix="Data Figure",
                    experiment_name=title.replace(":", ""),
                    fitness_metric=""
                )

            plt.show()

    @classmethod
    def plot_comorbidity_distribution(
            cls,
            df: pd.DataFrame,
            title: str = "Distribution of Comorbidity Scores",
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Plots the distribution of comorbidity scores using a severity gradient.
        (Migrated from DataAnalysis.ipynb)
        """
        # 1. Defensive Data Validation
        if 'Comorbidity' not in df.columns:
            raise ValueError("Data validation failed: 'Comorbidity' column is missing from the DataFrame.")

        # 2. Data Preparation
        counts = df['Comorbidity'].value_counts()

        if counts.empty:
            raise ValueError("The 'Comorbidity' column contains no valid data.")

        max_score = int(df['Comorbidity'].max())
        plot_data = counts.reindex(range(max_score + 1), fill_value=0)

        # 3. Invoke unified styling context
        with cls._style_context():
            fig, ax = plt.subplots(figsize=(10, 4.2), dpi=450)

            # 4. Rendering Engine
            sns.barplot(
                x=plot_data.index.astype(int),
                y=plot_data.values,
                hue=plot_data.index.astype(int),

                palette='rocket_r',

                ax=ax,
                edgecolor='black',
                width=0.6,
                linewidth=1.2,
                dodge=False,
                legend=False
            )

            # 5. Axes Formatting
            if title_on:
                ax.set_title(title, fontweight='bold', fontsize=14, pad=15)

            ax.set_xlabel('Comorbidity Score (Severity)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Patient Count', fontweight='bold', fontsize=11)

            # 6. Visual Tweaks
            sns.despine(ax=ax)

            max_count = int(plot_data.max())
            y_max = (max_count // 10 + 1) * 10

            ax.set_ylim(0, y_max)

            current_ticks = ax.get_yticks()
            new_ticks = [t for t in current_ticks if t < y_max] + [y_max]
            ax.set_yticks(new_ticks)

            for container in ax.containers:
                ax.bar_label(container, padding=4, color='black', fontweight='bold', fontsize=11)

            plt.tight_layout()

            # 7. Record and save logic (Streamlined: Fail Fast)
            if is_final_record:
                cls._save_figure(
                    fig=fig,
                    prefix="Data Figure",
                    experiment_name=title.replace(":", "").replace(" ", "_"),
                    fitness_metric=""
                )

            plt.show()

    @classmethod
    def plot_clinical_conditions_prevalence(
            cls,
            df: pd.DataFrame,
            title: str = "Prevalence of Clinical Conditions (Cohort Profile)",
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Plots the prevalence of clinical conditions.
        (Migrated from DataAnalysis.ipynb)
        """
        # 1. Defensive Data Validation
        if df.empty:
            raise ValueError("Data validation failed: The provided DataFrame is empty.")

        try:
            prevalence = df.sum(numeric_only=True).sort_values(ascending=False)
        except Exception as e:
            raise ValueError(f"Failed to calculate prevalence. Error: {e}")

        if prevalence.empty:
            raise ValueError("No valid numeric data found to calculate prevalence.")

        # 2. Invoke unified styling context
        with cls._style_context():
            fig, ax = plt.subplots(figsize=(10, 4.2), dpi=450)

            # 3. Rendering Engine (Vertical Bar Chart with Palette)
            sns.barplot(
                x=prevalence.index,
                y=prevalence.values,
                hue=prevalence.index,
                palette='viridis',
                edgecolor='black',
                width=0.6,
                linewidth=1.2,
                ax=ax,
                dodge=False,
                legend=False
            )

            # 4. Axes Formatting
            if title_on:
                ax.set_title(title, fontweight='bold', fontsize=14, pad=15)

            ax.set_xlabel('Clinical Condition', fontweight='bold', fontsize=11)
            ax.set_ylabel('Patient Count', fontweight='bold', fontsize=11)

            wrapped_labels = [
                textwrap.fill(str(label), width=12, break_long_words=False)
                for label in prevalence.index
            ]

            ax.set_xticks(ax.get_xticks())

            ax.set_xticklabels(wrapped_labels, rotation=0, ha='center')

            # 5. Visual Tweaks
            sns.despine(ax=ax)

            max_count = int(prevalence.max())

            magnitude = 10 ** (len(str(max_count)) - 1) if max_count >= 10 else 1
            if max_count // magnitude < 2 and magnitude >= 10:
                magnitude //= 10

            y_max = (max_count // magnitude + 1) * magnitude
            ax.set_ylim(0, y_max)

            current_ticks = ax.get_yticks()
            new_ticks = [t for t in current_ticks if t < y_max] + [y_max]
            ax.set_yticks(new_ticks)
            # ------------------------------------------

            for container in ax.containers:
                ax.bar_label(container, padding=3, color='black', fontweight='bold', fontsize=11)

            plt.tight_layout()

            # 6. Record and save logic
            if is_final_record:
                safe_name = title.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
                cls._save_figure(
                    fig=fig,
                    prefix="Data Figure",
                    experiment_name=safe_name,
                    fitness_metric=""
                )

            plt.show()

    @classmethod
    def plot_hfa_severity(
            cls,
            df: pd.DataFrame,
            title: str = "Hepatic Fat Accumulation (HFA) Severity",
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Plots the distribution of Hepatic Fat Accumulation severity.
        (Migrated from DataAnalysis.ipynb)
        """
        target_col = 'Hepatic Fat Accumulation (HFA)'

        # 1. Defensive Data Validation
        if target_col not in df.columns:
            raise ValueError(f"Data validation failed: '{target_col}' column is missing.")

        counts = df[target_col].value_counts()
        if counts.empty:
            raise ValueError(f"The '{target_col}' column contains no valid data.")

        max_score = int(df[target_col].max())
        plot_data = counts.reindex(range(max_score + 1), fill_value=0)

        # 2. Invoke unified styling context
        with cls._style_context():
            fig, ax = plt.subplots(figsize=(10, 4.2), dpi=450)

            # 3. Rendering Engine (Severity Gradient)
            sns.barplot(
                x=plot_data.index.astype(int),
                y=plot_data.values,
                hue=plot_data.index.astype(int),

                palette='YlOrRd',

                edgecolor='black',
                width=0.6,
                linewidth=1.2,
                ax=ax,
                dodge=False,
                legend=False
            )

            # 4. Axes Formatting (Direct Labeling on X-axis)
            if title_on:
                ax.set_title(title, fontweight='bold', fontsize=14, pad=15)

            ax.set_xlabel('Ultrasound Grade', fontweight='bold', fontsize=11)
            ax.set_ylabel('Patient Count', fontweight='bold', fontsize=11)

            grade_map = {
                0: "0\n(Normal)",
                1: "1\n(Mild)",
                2: "2\n(Moderate)",
                3: "3\n(Severe)",
                4: "4\n(Very Severe)"
            }

            ax.set_xticks(range(len(plot_data.index)))
            x_labels = [grade_map.get(int(idx), str(idx)) for idx in plot_data.index]
            ax.set_xticklabels(x_labels, fontsize=10)

            # 5. Visual Tweaks & Magnitude-Aware Capping
            sns.despine(ax=ax)

            max_count = int(plot_data.max())
            magnitude = 10 ** (len(str(max_count)) - 1) if max_count >= 10 else 1
            if max_count // magnitude < 2 and magnitude >= 10:
                magnitude //= 10

            y_max = (max_count // magnitude + 1) * magnitude
            ax.set_ylim(0, y_max)

            current_ticks = ax.get_yticks()
            new_ticks = [t for t in current_ticks if t < y_max] + [y_max]
            ax.set_yticks(new_ticks)

            for container in ax.containers:
                ax.bar_label(container, padding=3, color='black', fontweight='bold', fontsize=11)

            plt.tight_layout()

            # 6. Record and save logic
            if is_final_record:
                safe_name = title.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
                cls._save_figure(
                    fig=fig,
                    prefix="Data Figure",
                    experiment_name=safe_name,
                    fitness_metric=""
                )

            plt.show()

    @classmethod
    def plot_feature_distributions(
            cls,
            df: pd.DataFrame,
            columns: list = None,
            title: str = "Feature Distribution Analysis",
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Plots a dynamic grid of histograms with KDE overlays.
        (Academic Journal Standard)
        """
        if df.empty:
            raise ValueError("Data validation failed: The provided DataFrame is empty.")

        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        else:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            columns = df[columns].select_dtypes(include=['number']).columns.tolist()

        if not columns:
            raise ValueError("No valid numeric columns available to plot.")

        C_FILL = '#B3CDE3'  # Muted Pastel Blue (低饱和度填充，不抢戏)
        C_LINE = '#08306B'  # Deep Navy (高对比度边缘/KDE线)
        C_MEDIAN = '#D55E00'  # Vermilion (色弱友好的学术警示橙红)
        C_MEAN = '#333333'  # Charcoal (高级灰黑)

        n_features = len(columns)
        cols_count = min(3, n_features)
        rows_count = math.ceil(n_features / cols_count)

        with cls._style_context():
            fig, axes = plt.subplots(rows_count, cols_count, figsize=(15, 4 * rows_count + 2), dpi=450)

            if title_on:
                fig.suptitle(title, fontsize=16, y=1.02, fontweight='bold')

            axes_flat = np.array(axes).flatten() if n_features > 1 else [axes]

            for i, col_name in enumerate(columns):
                ax = axes_flat[i]

                valid_data = df[col_name].dropna()
                if valid_data.empty:
                    ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center', fontweight='bold')
                    continue

                mean_val = valid_data.mean()
                median_val = valid_data.median()

                if '(' in col_name and ')' in col_name:
                    clean_title = col_name.split('(')[0].strip()
                    unit_text = col_name[col_name.find('('):col_name.find(')') + 1]
                else:
                    clean_title = col_name
                    unit_text = "Value"

                # 渲染直方图
                sns.histplot(
                    data=df, x=col_name, kde=True, ax=ax,
                    color=C_FILL,
                    edgecolor=C_LINE,
                    alpha=0.7,
                    line_kws={'linewidth': 2}
                )

                # 强制劫持 KDE 曲线颜色，对齐边缘色
                if ax.lines:
                    ax.lines[0].set_color(C_LINE)

                # 渲染统计参考线
                ax.axvline(mean_val, color=C_MEAN, linestyle='--', linewidth=2.0,
                           label=f'Mean: {mean_val:.1f}', zorder=5)
                ax.axvline(median_val, color=C_MEDIAN, linestyle='-', linewidth=2.5,
                           label=f'Median: {median_val:.1f}', zorder=5)

                ax.set_title(clean_title, fontweight='bold', fontsize=12, pad=10)
                ax.set_xlabel(unit_text, fontweight='bold', fontsize=10)
                ax.set_ylabel('Frequency', fontweight='bold', fontsize=10)

                sns.despine(ax=ax)
                ax.legend(fontsize=9, loc='upper right', frameon=True, framealpha=0.95, edgecolor='black')

            for j in range(n_features, len(axes_flat)):
                fig.delaxes(axes_flat[j])

            plt.tight_layout()

            if is_final_record:
                safe_name = title.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
                cls._save_figure(fig=fig,
                                 prefix="Data Figure",
                                 experiment_name=safe_name, fitness_metric="")

            plt.show()

    @classmethod
    def plot_feature_outliers(
            cls,
            df: pd.DataFrame,
            columns: list = None,
            title: str = "Feature Outlier Analysis",
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Plots a dynamic grid of horizontal boxplots.
        (Academic Journal Standard - Color synchronized with distributions)
        """
        if df.empty:
            raise ValueError("Data validation failed: The provided DataFrame is empty.")

        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        else:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            columns = df[columns].select_dtypes(include=['number']).columns.tolist()

        if not columns:
            raise ValueError("No valid numeric columns available to plot.")

        C_FILL = '#B3CDE3'  # Muted Pastel Blue (主箱体)
        C_LINE = '#08306B'  # Deep Navy (箱体边缘与晶须)
        C_MEDIAN = '#D55E00'  # Vermilion (中位数，与上方分布图的实线完全呼应)
        C_OUTLIER = '#CB181D'  # Crimson (异常值专用深红色)


        n_features = len(columns)
        cols_count = min(3, n_features)
        rows_count = math.ceil(n_features / cols_count)

        with cls._style_context():
            fig, axes = plt.subplots(rows_count, cols_count, figsize=(15, 4 * rows_count + 2), dpi=450)

            if title_on:
                fig.suptitle(title, fontsize=16, y=1.02, fontweight='bold')

            axes_flat = np.array(axes).flatten() if n_features > 1 else [axes]

            for i, col_name in enumerate(columns):
                ax = axes_flat[i]

                if '(' in col_name and ')' in col_name:
                    clean_title = col_name.split('(')[0].strip()
                    unit_text = col_name[col_name.find('('):col_name.find(')') + 1]
                else:
                    clean_title = col_name
                    unit_text = "Value"

                # 渲染箱线图 (深度定制属性)
                sns.boxplot(
                    data=df, x=col_name, ax=ax,
                    color=C_FILL,
                    linewidth=1.5,
                    linecolor=C_LINE,

                    flierprops={
                        "marker": "o",
                        "markerfacecolor": "none",  # 空心
                        "markeredgecolor": C_OUTLIER,
                        "markersize": 5,
                        "alpha": 0.8,
                        "markeredgewidth": 1.2
                    },

                    # 中位数：精确呼应分布图的 Vermilion 橙红色
                    medianprops={"color": C_MEDIAN, "linewidth": 2.5}
                )

                ax.set_title(clean_title, fontsize=12, fontweight='bold', pad=10)
                ax.set_xlabel(unit_text, fontsize=10, fontweight='bold')
                ax.set_ylabel('')

                sns.despine(ax=ax, left=True)
                ax.set_yticks([])
                ax.grid(axis='x', linestyle='--', alpha=0.5)

            for j in range(n_features, len(axes_flat)):
                fig.delaxes(axes_flat[j])

            plt.tight_layout()

            if is_final_record:
                safe_name = title.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
                cls._save_figure(fig=fig,
                                 prefix="Data Figure",
                                 experiment_name=safe_name,
                                 fitness_metric="")

            plt.show()

    @classmethod
    def plot_target_correlation(
            cls,
            df: pd.DataFrame,
            target_col: str = 'Gallstone Status',
            title: str = "Global Feature Correlation",
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Plots the Pearson correlation of all numeric features against a specific target.
        (Academic Journal Standard - Synced Palette & Ascending Order)
        """
        # 1. Defensive Data Validation
        if df.empty:
            raise ValueError("Data validation failed: The provided DataFrame is empty.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        if not pd.api.types.is_numeric_dtype(df[target_col]):
            try:
                target_series = pd.to_numeric(df[target_col])
            except ValueError:
                raise TypeError(f"Target column '{target_col}' must be numeric to compute correlation.")
        else:
            target_series = df[target_col]

        # 2. Data Pipeline
        df_numeric = df.select_dtypes(include=['number', 'bool']).drop(columns=[target_col], errors='ignore')

        if df_numeric.empty:
            raise ValueError("No numeric features available to compute correlation.")

        target_corr = df_numeric.corrwith(target_series).dropna().sort_values(ascending=True)

        if target_corr.empty:
            raise ValueError("Correlation computation yielded no valid results.")

        # 3. Invoke unified styling context
        with cls._style_context():

            # 略微增加行高乘数 (从 0.3 提升到 0.35)，为特征名称留出更多垂直呼吸空间
            fig_height = max(8.0, len(target_corr) * 0.35)
            fig, ax = plt.subplots(figsize=(10, fig_height), dpi=450)

            C_POS = '#D55E00'
            C_NEG = '#08306B'
            colors = [C_POS if x > 0 else C_NEG for x in target_corr.values]

            ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor='black', linewidth=0.8, height=0.55)

            # 4. Axes Formatting
            if title_on:
                ax.set_title(f"{title} with {target_col}", fontweight='bold', fontsize=14, pad=15)

            ax.set_xlabel("Pearson Correlation Coefficient (r)", fontweight='bold', fontsize=11)

            ax.set_ylabel('')

            ax.tick_params(axis='y', labelsize=9, labelcolor='#333333')

            # 5. Visual Tweaks
            sns.despine(ax=ax, left=True)

            # 零位基准线
            ax.axvline(0, color='black', linewidth=1.2, zorder=3)

            # 底层辅助网格
            ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)

            ax.invert_yaxis()

            plt.tight_layout()

            # 6. Record and save logic
            if is_final_record:
                safe_name = title.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
                cls._save_figure(
                    fig=fig,
                    prefix="Data Figure",
                    experiment_name=safe_name,
                    fitness_metric=""
                )

            plt.show()

    @classmethod
    def plot_bmi_repair_bland_altman(
            cls,
            df_raw  : pd.DataFrame,
            df_fixed: pd.DataFrame,
            main_title : str = None,
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Specialized Bland-Altman plot for BMI Repair Analysis.
        Ultimate Optimization: Zero-interference layout, edge-aligned metadata,
        and distinct hollow markers for raw vs repaired states.
        """

        df_raw = df_raw.reset_index(drop=True)
        df_fixed = df_fixed.reset_index(drop=True)

        calc_raw = df_raw['Weight'] / ((df_raw['Height'] / 100) ** 2)
        calc_fixed = df_fixed['Weight'] / ((df_fixed['Height'] / 100) ** 2)

        plot_df = pd.DataFrame()
        plot_df['Mean_Raw'] = (df_raw['Body Mass Index (BMI)'] + calc_raw) / 2.0
        plot_df['Diff_Raw'] = df_raw['Body Mass Index (BMI)'] - calc_raw

        plot_df['Mean_Fixed'] = (df_fixed['Body Mass Index (BMI)'] + calc_fixed) / 2.0
        plot_df['Diff_Fixed'] = df_fixed['Body Mass Index (BMI)'] - calc_fixed

        is_repaired = ~np.isclose(df_raw['Body Mass Index (BMI)'], df_fixed['Body Mass Index (BMI)'], equal_nan=True)
        plot_df['Status'] = np.where(is_repaired, 'Repaired', 'Valid')
        plot_df = plot_df.dropna()

        mask_valid = plot_df['Status'] == 'Valid'
        mask_repaired = plot_df['Status'] == 'Repaired'

        # --- 统一视觉词汇表 ---
        C_VALID = '#08306B'  # Deep Navy (合法数据)
        C_OUTLIER = '#D55E00'  # Vermilion (原始异常值)
        C_REPAIRED = '#27AE60'  # Green (修复后数据)

        def add_background_kde(ax, mean_series, diff_series, color):
            diff_array = diff_series.values.astype(float)
            if diff_array.std() < 1e-4:
                diff_array = diff_array + np.random.normal(0, 1e-3, len(diff_array))
            kde = stats.gaussian_kde(diff_array)
            y_vals = np.linspace(diff_array.min() - 0.5, diff_array.max() + 0.5, 300)
            density = kde(y_vals)
            x_min, x_max = mean_series.min(), mean_series.max()
            scaled_density = x_min + (density / density.max()) * ((x_max - x_min) * 0.3)
            ax.fill_betweenx(y_vals, x_min, scaled_density, color=color, alpha=0.15, zorder=0, linewidth=0)
            ax.plot(scaled_density, y_vals, color=color, alpha=0.3, zorder=0, lw=1)

        SUBPLOT_TITLE_KWS = {
            'loc': 'left',
            'fontstyle': 'italic',
            'fontweight': 'normal',
            'fontsize': 13,
            'color': '#444444',
            'pad': 15
        }

        with cls._style_context():
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True, dpi=450)
            ax_raw, ax_fix = axes[0], axes[1]

            # ==========================================
            # A. Before Repair (Raw)
            # ==========================================
            add_background_kde(ax_raw, plot_df['Mean_Raw'], plot_df['Diff_Raw'], color=C_OUTLIER)

            # 健康数据 (深蓝实心)
            ax_raw.scatter(plot_df.loc[mask_valid, 'Mean_Raw'], plot_df.loc[mask_valid, 'Diff_Raw'],
                           c=C_VALID, alpha=0.4, edgecolors='white', lw=0.5, s=60, zorder=3)
            # 异常数据 (橙红空心)
            ax_raw.scatter(plot_df.loc[mask_repaired, 'Mean_Raw'], plot_df.loc[mask_repaired, 'Diff_Raw'],
                           facecolors='none', edgecolors=C_OUTLIER, lw=1.5, s=60, zorder=4)

            m_r, sd_r = plot_df['Diff_Raw'].mean(), plot_df['Diff_Raw'].std()
            ax_raw.axhline(m_r, color='black', ls='-', lw=2, zorder=2)
            ax_raw.axhline(m_r + 1.96 * sd_r, color='gray', ls='--', lw=1.5, zorder=2)
            ax_raw.axhline(m_r - 1.96 * sd_r, color='gray', ls='--', lw=1.5, zorder=2)

            # [终极优化] LoA 极简文本放在右下角内部 (相对坐标 x=0.98, y=0.03)
            ax_raw.text(0.98, 0.03, f'LoA: {sd_r:.2f}', transform=ax_raw.transAxes,
                        color=C_OUTLIER, ha='right', va='bottom', fontweight='bold', fontsize=12, zorder=5)

            ax_raw.set_title("(a) Before Repair", **SUBPLOT_TITLE_KWS)
            ax_raw.set_xlabel("Mean BMI (Original)", fontweight='bold')
            ax_raw.set_ylabel("Difference (Reported - Calculated)", fontweight='bold')

            # ==========================================
            # B. After Repair (Fixed)
            # ==========================================
            add_background_kde(ax_fix, plot_df['Mean_Fixed'], plot_df['Diff_Fixed'], color=C_REPAIRED)

            # 健康数据 (深蓝实心)
            ax_fix.scatter(plot_df.loc[mask_valid, 'Mean_Fixed'], plot_df.loc[mask_valid, 'Diff_Fixed'],
                           c=C_VALID, alpha=0.4, edgecolors='white', lw=0.5, s=60, zorder=3)
            # 修复后数据 (绿色空心)
            ax_fix.scatter(plot_df.loc[mask_repaired, 'Mean_Fixed'], plot_df.loc[mask_repaired, 'Diff_Fixed'],
                           facecolors='none', edgecolors=C_REPAIRED, lw=1.5, s=60, zorder=4)

            m_f, sd_f = plot_df['Diff_Fixed'].mean(), plot_df['Diff_Fixed'].std() if plot_df[
                                                                                         'Diff_Fixed'].std() > 0 else 0.001
            ax_fix.axhline(m_f, color='black', ls='-', lw=2, zorder=2)
            ax_fix.axhline(m_f + 1.96 * sd_f, color='gray', ls='--', lw=1.5, zorder=2)
            ax_fix.axhline(m_f - 1.96 * sd_f, color='gray', ls='--', lw=1.5, zorder=2)

            # [终极优化] LoA 极简文本放在右下角内部
            ax_fix.text(0.98, 0.03, f'LoA: {sd_f:.2f}', transform=ax_fix.transAxes,
                        color=C_REPAIRED, ha='right', va='bottom', fontweight='bold', fontsize=12, zorder=5)

            ax_fix.set_title("(b) After Repair", **SUBPLOT_TITLE_KWS)
            ax_fix.set_xlabel("Mean BMI (Repaired)", fontweight='bold')
            ax_fix.set_ylabel('')
            ax_fix.tick_params(left=False)


            custom_lines = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C_VALID, markersize=8, alpha=0.6,
                       label='Valid Data'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_OUTLIER, markerfacecolor='none', markersize=8,
                       markeredgewidth=1.5, label='Outliers (Raw)'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_REPAIRED, markerfacecolor='none',
                       markersize=8, markeredgewidth=1.5, label='Repaired Data')
            ]

            fig.legend(handles=custom_lines,
                       loc='lower center',
                       bbox_to_anchor=(0.5, 0.02),
                       ncol=3,
                       frameon=False,
                       fontsize=11)

            plt.tight_layout(rect=[0, 0.08, 1, 0.96])

            sns.despine(ax=ax_raw)
            sns.despine(ax=ax_fix, left=True)

            if title_on:
                fig.suptitle("Bland-Altman Agreement Analysis", y=1.02, fontsize=16,
                             fontweight='bold')

            plt.tight_layout(rect=[0, 0.1, 1, 0.98])

            if is_final_record:
                cls._save_figure(fig=fig,
                                 prefix="Data Figure",
                                 experiment_name=main_title,
                                 fitness_metric="")

            plt.show()

    @classmethod
    def plot_4track_repair_impact(
            cls,
            df_raw: pd.DataFrame,
            df_fixed: pd.DataFrame,
            track_config: dict,
            main_title: str = None,
            is_final_record: bool = False
    ):
        """
        Pure geometric A/B comparison map for 4-Track Physics Logic.
        Shows Raw data on the left and Fixed data on the right, mapped to the same absolute limits.
        """

        x_col = track_config['x_col']
        y_col = track_config['y_col']

        lim_a = track_config['lim_a']
        lim_b = track_config['lim_b']
        lim_d = track_config['lim_d']

        max_val = max(
            df_raw[x_col].abs().max(),
            df_raw[y_col].abs().max(),
            df_fixed[x_col].abs().max(),
            df_fixed[y_col].abs().max(),
            lim_a, lim_b, lim_d
        ) * 1.15

        SUBPLOT_TITLE_KWS = {
            'loc': 'left',
            'fontstyle': 'italic',
            'fontweight': 'normal',
            'fontsize': 13,
            'color': '#444444',
            'pad': 15
        }

        with cls._style_context():
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True, dpi=450)
            ax_raw, ax_fix = axes[0], axes[1]

            def draw_geometric_tracks(ax):
                # Track A (垂直轨道 - 橙色)
                ax.axvline(-lim_a, color='#F39C12', ls='-', lw=2.5, alpha=0.8, label=f'Track A (Sum: ±{lim_a:.2f})')
                ax.axvline(lim_a, color='#F39C12', ls='-', lw=2.5, alpha=0.8)

                # Track B (水平轨道 - 紫色)
                ax.axhline(-lim_b, color='#9B59B6', ls='-', lw=2.5, alpha=0.8, label=f'Track B (Ratio: ±{lim_b:.2f})')
                ax.axhline(lim_b, color='#9B59B6', ls='-', lw=2.5, alpha=0.8)

                # Track D (对角线轨道 - 绿色)
                diag_x = np.linspace(-max_val * 2, max_val * 2, 200)
                ax.plot(diag_x, diag_x - lim_d, color='#27AE60', ls='-', lw=2.5, alpha=0.8,
                        label=f'Track D (Diag: ±{lim_d:.2f})')
                ax.plot(diag_x, diag_x + lim_d, color='#27AE60', ls='-', lw=2.5, alpha=0.8)

                # 零位十字基准线
                ax.axhline(0, color='black', linewidth=1, alpha=0.4, zorder=1)
                ax.axvline(0, color='black', linewidth=1, alpha=0.4, zorder=1)

                ax.set_aspect('equal')
                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)
                ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

            draw_geometric_tracks(ax_raw)
            ax_raw.scatter(df_raw[x_col], df_raw[y_col], facecolors='none', edgecolors='#08306B',
                           s=50, lw=1.2, alpha=0.6, zorder=5)

            ax_raw.set_title("(a) Before Repair", **SUBPLOT_TITLE_KWS)
            ax_raw.set_xlabel(f"Track A Deviation (Summation)", fontweight='bold')
            ax_raw.set_ylabel(f"Track B Deviation (Ratio)", fontweight='bold')

            draw_geometric_tracks(ax_fix)
            ax_fix.scatter(df_fixed[x_col], df_fixed[y_col], facecolors='none', edgecolors='#08306B',
                           s=50, lw=1.2, alpha=0.6, zorder=5)

            ax_fix.set_title("(b) After Repair", **SUBPLOT_TITLE_KWS)
            ax_fix.set_xlabel(f"Track A Deviation (Summation)", fontweight='bold')
            ax_fix.tick_params(left=False)

            sns.despine(ax=ax_raw)
            sns.despine(ax=ax_fix, left=True)

            handles, labels = ax_fix.get_legend_handles_labels()
            unique_dict = dict(zip(labels, handles))

            fig.legend(unique_dict.values(), unique_dict.keys(), loc='lower center',
                       bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=11)

            fig.suptitle(main_title, y=0.98, fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0.12, 1, 0.95])

            if is_final_record:
                cls._save_figure(fig=fig, prefix="Data Figure", experiment_name=main_title, fitness_metric="")

            plt.show()

    @classmethod
    def plot_2track_repair_impact(
            cls,
            df_raw: pd.DataFrame,
            df_fixed: pd.DataFrame,
            track_config: dict,
            main_title: str = None,
            is_final_record: bool = False
    ):
        """
        Pure geometric A/B comparison map for orthogonal 2-Track Physics Logic.
        Dynamically configurable for any two absolute-volume tracks (e.g., Fat, Muscle).
        """

        x_col = track_config['x_col']
        y_col = track_config['y_col']
        lim_x = track_config['lim_x']
        lim_y = track_config['lim_y']

        max_val = max(
            df_raw[x_col].abs().max(),
            df_raw[y_col].abs().max(),
            df_fixed[x_col].abs().max(),
            df_fixed[y_col].abs().max(),
            lim_x, lim_y
        ) * 1.15

        SUBPLOT_TITLE_KWS = {
            'loc': 'left',
            'fontstyle': 'italic',
            'fontweight': 'normal',
            'fontsize': 13,
            'color': '#444444',
            'pad': 15
        }

        with cls._style_context():
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True, dpi=450)
            ax_raw, ax_fix = axes[0], axes[1]

            def draw_orthogonal_tracks(ax):
                # 垂直轨道 (X)
                ax.axvline(-lim_x, color='#F39C12', ls='-', lw=2.5, alpha=0.8, label=f"Track X Limit: ±{lim_x:.2f}")
                ax.axvline(lim_x, color='#F39C12', ls='-', lw=2.5, alpha=0.8)

                # 水平轨道 (Y)
                ax.axhline(-lim_y, color='#9B59B6', ls='-', lw=2.5, alpha=0.8, label=f"Track Y Limit: ±{lim_y:.2f}")
                ax.axhline(lim_y, color='#9B59B6', ls='-', lw=2.5, alpha=0.8)

                ax.axhline(0, color='black', linewidth=1, alpha=0.4, zorder=1)
                ax.axvline(0, color='black', linewidth=1, alpha=0.4, zorder=1)

                ax.set_aspect('equal')
                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)
                ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

            # --- A. Raw Data ---
            draw_orthogonal_tracks(ax_raw)
            ax_raw.scatter(df_raw[x_col], df_raw[y_col], facecolors='none', edgecolors='#08306B',
                           s=50, lw=1.2, alpha=0.6, zorder=5)

            ax_raw.set_title("(a) Before Repair: Raw Distribution", **SUBPLOT_TITLE_KWS)
            ax_raw.set_xlabel(track_config.get('label_x', x_col), fontweight='bold')
            ax_raw.set_ylabel(track_config.get('label_y', y_col), fontweight='bold')

            # --- B. Fixed Data ---
            draw_orthogonal_tracks(ax_fix)
            ax_fix.scatter(df_fixed[x_col], df_fixed[y_col], facecolors='none', edgecolors='#08306B',
                           s=50, lw=1.2, alpha=0.6, zorder=5)

            ax_fix.set_title("(b) After Repair: Resolved Manifold", **SUBPLOT_TITLE_KWS)
            ax_fix.set_xlabel(track_config.get('label_x', x_col), fontweight='bold')
            ax_fix.tick_params(left=False)

            # --- Formatting ---
            sns.despine(ax=ax_raw)
            sns.despine(ax=ax_fix, left=True)

            handles, labels = ax_fix.get_legend_handles_labels()
            unique_dict = dict(zip(labels, handles))
            fig.legend(unique_dict.values(), unique_dict.keys(), loc='lower center',
                       bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=11)

            main_title = track_config.get('title', "2-Track Physics Diagnostic Map")
            fig.suptitle(f"{main_title}", y=0.98, fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0.12, 1, 0.95])

            if is_final_record:
                cls._save_figure(fig=fig, prefix="Data Figure", experiment_name=main_title, fitness_metric="")

            plt.show()

    @classmethod
    def plot_empirical_cone_repair(
            cls,
            df_raw: pd.DataFrame,
            df_fixed: pd.DataFrame,
            x_col: str,
            y_col: str,
            main_title: str = None,
            is_final_record: bool = False
    ):
        """
        Visualizes empirical statistical repairs (ratios).
        Draws a 'Confidence Cone' based on the 2.5th and 97.5th percentiles of the ratio.
        """
        raw_ratio = df_raw[y_col] / df_raw[x_col]

        q_low = raw_ratio.quantile(0.025)
        q_high = raw_ratio.quantile(0.975)
        q_median = raw_ratio.median()

        is_repaired = (~np.isclose(df_raw[x_col], df_fixed[x_col], equal_nan=True)) | \
                      (~np.isclose(df_raw[y_col], df_fixed[y_col], equal_nan=True))

        max_x = max(df_raw[x_col].max(), df_fixed[x_col].max()) * 1.05
        min_x = max(0, min(df_raw[x_col].min(), df_fixed[x_col].min()) * 0.95)

        SUBPLOT_TITLE_KWS = {
            'loc': 'left',
            'fontstyle': 'italic',
            'fontweight': 'normal',
            'fontsize': 13,
            'color': '#444444',
            'pad': 15
        }

        C_VALID = '#08306B'  # Deep Navy
        C_OUTLIER = '#D55E00'  # Vermilion
        C_REPAIRED = '#27AE60'  # Green

        with cls._style_context():
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True, dpi=450)
            ax_raw, ax_fix = axes[0], axes[1]

            def draw_confidence_cone(ax):
                # 生成绘制圆锥的 X 轴基准线
                x_vals = np.linspace(0, max_x * 1.1, 100)

                # 绘制 95% 安全锥的阴影区域
                ax.fill_between(x_vals, x_vals * q_low, x_vals * q_high,
                                color='#BDC3C7', alpha=0.3, zorder=0, label='95% Confidence Cone')

                # 绘制中位数基准线
                ax.plot(x_vals, x_vals * q_median, color='#34495E', ls='--', lw=2, alpha=0.8, zorder=1,
                        label=f'Median Ratio ({q_median:.3f})')

                # 绘制锥体边界
                ax.plot(x_vals, x_vals * q_low, color='#7F8C8D', ls=':', lw=1.5, alpha=0.6)
                ax.plot(x_vals, x_vals * q_high, color='#7F8C8D', ls=':', lw=1.5, alpha=0.6)

                ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
                ax.set_xlim(min_x, max_x)

                p99_y = df_raw[y_col].quantile(0.99)
                ax.set_ylim(0, max(p99_y * 1.5, (max_x * q_high) * 1.1))

            draw_confidence_cone(ax_raw)

            ax_raw.scatter(df_raw.loc[~is_repaired, x_col], df_raw.loc[~is_repaired, y_col],
                           c=C_VALID, alpha=0.4, edgecolors='white', lw=0.5, s=60, zorder=3)

            ax_raw.scatter(df_raw.loc[is_repaired, x_col], df_raw.loc[is_repaired, y_col],
                           facecolors='none', edgecolors=C_OUTLIER, lw=1.5, s=60, zorder=4)

            ax_raw.set_title("(a) Before Repair: Empirical Distribution", **SUBPLOT_TITLE_KWS)
            ax_raw.set_xlabel(x_col, fontweight='bold')
            ax_raw.set_ylabel(y_col, fontweight='bold')

            draw_confidence_cone(ax_fix)

            # 正常数据
            ax_fix.scatter(df_fixed.loc[~is_repaired, x_col], df_fixed.loc[~is_repaired, y_col],
                           c=C_VALID, alpha=0.4, edgecolors='white', lw=0.5, s=60, zorder=3)
            # 修复数据 (被吸回锥体内部)
            ax_fix.scatter(df_fixed.loc[is_repaired, x_col], df_fixed.loc[is_repaired, y_col],
                           facecolors='none', edgecolors=C_REPAIRED, lw=1.5, s=60, zorder=4)

            ax_fix.set_title("(b) After Repair: Statistical Re-alignment", **SUBPLOT_TITLE_KWS)
            ax_fix.set_xlabel(x_col, fontweight='bold')
            ax_fix.tick_params(left=False)

            sns.despine(ax=ax_raw)
            sns.despine(ax=ax_fix, left=True)

            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='#BDC3C7', lw=6, alpha=0.5, label='95% Safe Zone (2.5% - 97.5%)'),
                Line2D([0], [0], color='#34495E', ls='--', lw=2, label='Median Empirical Truth'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C_VALID, markersize=8, alpha=0.6,
                       label='Valid Data'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_OUTLIER, markerfacecolor='none', markersize=8,
                       markeredgewidth=1.5, label='Outliers (Raw)'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_REPAIRED, markerfacecolor='none',
                       markersize=8, markeredgewidth=1.5, label='Repaired Data')
            ]

            fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False,
                       fontsize=10)
            fig.suptitle(main_title, y=0.98, fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])

            if is_final_record:
                cls._save_figure(fig=fig, prefix="Data Figure", experiment_name=main_title, fitness_metric="")

            plt.show()