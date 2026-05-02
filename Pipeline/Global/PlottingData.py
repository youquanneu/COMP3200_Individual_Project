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
                height=3,
                aspect=1.3,
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
                g.figure.subplots_adjust(bottom=0.10)

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
            fig, ax = plt.subplots(figsize=(10, 6), dpi=450)

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
            fig, ax = plt.subplots(figsize=(10, 6), dpi=450)

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
            fig, ax = plt.subplots(figsize=(10, 6), dpi=450)

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
                0: "No Fat Accumulation",
                1: "Mild",
                2: "Moderate",
                3: "Severe",
                4: "Very Severe"
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

            abs_corr = target_corr.abs()
            max_abs = abs_corr.max() if abs_corr.max() != 0 else 1.0  # 防止除零

            alphas = 0.5 + (abs_corr / max_abs) * 0.5

            from matplotlib.colors import to_rgba
            bar_colors = [
                to_rgba(C_POS, alpha=a) if v > 0 else to_rgba(C_NEG, alpha=a)
                for v, a in zip(target_corr.values, alphas)
            ]

            ax.barh(target_corr.index, target_corr.values, color=bar_colors,
                    edgecolor='black', linewidth=0.6, height=0.65)

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
            df_raw: pd.DataFrame,
            df_fixed: pd.DataFrame,
            main_title: str = None,
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Specialized Bland-Altman plot for BMI Repair Analysis (Publication Ready).
        Ultimate Optimization: Zero-interference layout, edge-aligned metadata,
        and giant hollow markers for raw vs corrected states.
        """
        df_raw = df_raw.reset_index(drop=True)
        df_fixed = df_fixed.reset_index(drop=True)

        # Calculated values based on fundamental physical traits
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
        C_CORRECTED = '#27AE60'  # Green (修正后数据)

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
            'loc': 'left', 'fontstyle': 'italic', 'fontweight': 'normal',
            'fontsize': 13, 'color': '#444444', 'pad': 15
        }

        with cls._style_context():
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True, dpi=450)
            ax_raw, ax_fix = axes[0], axes[1]

            # ==========================================
            # A. Raw (Before Correction)
            # ==========================================
            add_background_kde(ax_raw, plot_df['Mean_Raw'], plot_df['Diff_Raw'], color=C_OUTLIER)

            # 健康数据 (深蓝实心)
            ax_raw.scatter(plot_df.loc[mask_valid, 'Mean_Raw'], plot_df.loc[mask_valid, 'Diff_Raw'],
                           c=C_VALID, alpha=0.5, edgecolors='white', lw=0.5, s=60, zorder=3)
            # 异常数据 (橙红巨型空心圈)
            ax_raw.scatter(plot_df.loc[mask_repaired, 'Mean_Raw'], plot_df.loc[mask_repaired, 'Diff_Raw'],
                           facecolors='none', edgecolors=C_OUTLIER, lw=2.5, s=200, alpha=0.8, zorder=4)

            m_r, sd_r = plot_df['Diff_Raw'].mean(), plot_df['Diff_Raw'].std()
            ax_raw.axhline(m_r, color='black', ls='-', lw=2, zorder=2)
            ax_raw.axhline(m_r + 1.96 * sd_r, color='gray', ls='--', lw=1.5, zorder=2)
            ax_raw.axhline(m_r - 1.96 * sd_r, color='gray', ls='--', lw=1.5, zorder=2)

            ax_raw.text(0.98, 0.03, f'LoA: {sd_r:.2f}', transform=ax_raw.transAxes,
                        color=C_OUTLIER, ha='right', va='bottom', fontweight='bold', fontsize=12, zorder=5)

            ax_raw.set_title("(a) Raw (BMI Consistency)", **SUBPLOT_TITLE_KWS)
            # 学术术语对齐: Observed vs Calculated
            ax_raw.set_xlabel("Mean BMI (Observed & Calculated)", fontweight='bold')
            ax_raw.set_ylabel("Difference (Observed - Calculated)", fontweight='bold')

            # ==========================================
            # B. Corrected Data
            # ==========================================
            add_background_kde(ax_fix, plot_df['Mean_Fixed'], plot_df['Diff_Fixed'], color=C_CORRECTED)

            # 健康数据 (深蓝实心)
            ax_fix.scatter(plot_df.loc[mask_valid, 'Mean_Fixed'], plot_df.loc[mask_valid, 'Diff_Fixed'],
                           c=C_VALID, alpha=0.5, edgecolors='white', lw=0.5, s=60, zorder=3)
            # 修正后数据 (翠绿巨型空心圈)
            ax_fix.scatter(plot_df.loc[mask_repaired, 'Mean_Fixed'], plot_df.loc[mask_repaired, 'Diff_Fixed'],
                           facecolors='none', edgecolors=C_CORRECTED, lw=2.5, s=200, alpha=0.8, zorder=4)

            m_f, sd_f = plot_df['Diff_Fixed'].mean(), plot_df['Diff_Fixed'].std() if plot_df[
                                                                                         'Diff_Fixed'].std() > 0 else 0.001
            ax_fix.axhline(m_f, color='black', ls='-', lw=2, zorder=2)
            ax_fix.axhline(m_f + 1.96 * sd_f, color='gray', ls='--', lw=1.5, zorder=2)
            ax_fix.axhline(m_f - 1.96 * sd_f, color='gray', ls='--', lw=1.5, zorder=2)

            ax_fix.text(0.98, 0.03, f'LoA: {sd_f:.2f}', transform=ax_fix.transAxes,
                        color=C_CORRECTED, ha='right', va='bottom', fontweight='bold', fontsize=12, zorder=5)

            ax_fix.set_title("(b) Corrected (BMI Consistency)", **SUBPLOT_TITLE_KWS)
            # 学术术语对齐
            ax_fix.set_xlabel("Mean BMI (Observed & Calculated)", fontweight='bold')
            ax_fix.set_ylabel('')
            ax_fix.tick_params(left=False)

            # --- 极简统一图例 ---
            custom_lines = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C_VALID, markersize=8, alpha=0.6,
                       label='Valid Data'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_OUTLIER, markerfacecolor='none',
                       markersize=12,
                       markeredgewidth=2, label='Outliers (Raw)'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_CORRECTED, markerfacecolor='none',
                       markersize=12, markeredgewidth=2, label='Corrected Data')
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
                fig.suptitle(main_title if main_title else "BMI Consistency Analysis", y=1.02, fontsize=16,
                             fontweight='bold')

            plt.tight_layout(rect=[0, 0.1, 1, 0.98])

            # --- 防弹保存逻辑 ---
            if is_final_record:
                safe_name = main_title.replace(":", "").replace(" ", "_").replace("(", "").replace(")",
                                                                                                   "") if main_title else "BMI_Bland_Altman"
                cls._save_figure(fig=fig,
                                 prefix="Data Figure",
                                 experiment_name=safe_name,
                                 fitness_metric="")

            plt.show()

    @classmethod
    def plot_4track_repair_impact(
            cls,
            df_raw: pd.DataFrame,
            df_fixed: pd.DataFrame,
            track_config: dict,
            main_title: str = None,
            is_final_record: bool = False,
            title_on: bool = True
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
            fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True, dpi=450)
            ax_raw, ax_fix = axes[0], axes[1]

            def draw_geometric_tracks(ax):
                # Track A (垂直轨道 - 橙色)
                ax.axvline(-lim_a, color='#F39C12', ls='-', lw=2.5, alpha=0.8,
                           label=rf'$\tau_{{TBW}} \text{{ (Track A)}}: \pm{lim_a:.2f}$')
                ax.axvline(lim_a, color='#F39C12', ls='-', lw=2.5, alpha=0.8)

                # Track B (水平轨道 - 紫色)
                ax.axhline(-lim_b, color='#9B59B6', ls='-', lw=2.5, alpha=0.8,
                           label=rf'$\tau_{{TBW}} \text{{ (Track B)}}: \pm{lim_b:.2f}$')
                ax.axhline(lim_b, color='#9B59B6', ls='-', lw=2.5, alpha=0.8)

                # Track D (对角线轨道 - 绿色)
                diag_x = np.linspace(-max_val * 2, max_val * 2, 200)
                ax.plot(diag_x, diag_x - lim_d, color='#27AE60', ls='-', lw=2.5, alpha=0.8,
                        label=rf'$\tau_{{TBW}} \text{{ (Track D)}}: \pm{lim_d:.2f}$')
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
            ax_raw.set_xlabel(r"$\varepsilon_{TBW} \text{ (Track A)}$", fontweight='bold', fontsize=14)
            ax_raw.set_ylabel(r"$\varepsilon_{TBW} \text{ (Track B)}$", fontweight='bold', fontsize=14)

            draw_geometric_tracks(ax_fix)
            ax_fix.scatter(df_fixed[x_col], df_fixed[y_col], facecolors='none', edgecolors='#08306B',
                           s=50, lw=1.2, alpha=0.6, zorder=5)

            ax_fix.set_title("(b) After Repair", **SUBPLOT_TITLE_KWS)
            ax_fix.set_xlabel(r"$\varepsilon_{TBW} \text{ (Track A)}$", fontweight='bold', fontsize=14)
            ax_fix.tick_params(left=False)

            sns.despine(ax=ax_raw)
            sns.despine(ax=ax_fix, left=True)

            handles, labels = ax_fix.get_legend_handles_labels()
            unique_dict = dict(zip(labels, handles))

            fig.legend(unique_dict.values(), unique_dict.keys(), loc='lower center',
                       bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=11)

            if title_on:
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
            is_final_record: bool = False,
            title_on: bool = True
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
            fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True, dpi=450)
            ax_raw, ax_fix = axes[0], axes[1]

            def draw_orthogonal_tracks(ax):
                # 垂直轨道 (X)
                ax.axvline(-lim_x, color='#F39C12', ls='-', lw=2.5, alpha=0.8,
                           label=rf"$\tau_{{TFC}} \text{{ (Track E)}}: \pm{lim_x:.2f}$")
                ax.axvline(lim_x, color='#F39C12', ls='-', lw=2.5, alpha=0.8)

                # 水平轨道 (Y)
                ax.axhline(-lim_y, color='#9B59B6', ls='-', lw=2.5, alpha=0.8,
                           label=rf"$\tau_{{TFC}} \text{{ (Track F)}}: \pm{lim_y:.2f}$")
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

            ax_raw.set_title("(a) Raw", **SUBPLOT_TITLE_KWS)
            ax_raw.set_xlabel(r"$\varepsilon_{TFC} \text{ (Track E)}$", fontweight='bold', fontsize=14)
            ax_raw.set_ylabel(r"$\varepsilon_{TFC} \text{ (Track F)}$", fontweight='bold', fontsize=14)

            # --- B. Fixed Data ---
            draw_orthogonal_tracks(ax_fix)
            ax_fix.scatter(df_fixed[x_col], df_fixed[y_col], facecolors='none', edgecolors='#08306B',
                           s=50, lw=1.2, alpha=0.6, zorder=5)

            ax_fix.set_title("(b) Corrected", **SUBPLOT_TITLE_KWS)
            ax_fix.set_xlabel(r"$\varepsilon_{TFC} \text{ (Track E)}$", fontweight='bold', fontsize=14)
            ax_fix.tick_params(left=False)

            # --- Formatting ---
            sns.despine(ax=ax_raw)
            sns.despine(ax=ax_fix, left=True)

            handles, labels = ax_fix.get_legend_handles_labels()
            unique_dict = dict(zip(labels, handles))
            fig.legend(unique_dict.values(), unique_dict.keys(), loc='lower center',
                       bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=11)
            if title_on:
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
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        Visualizes empirical statistical repairs (ratios).
        Draws a 'Confidence Cone' based on the 2.5th and 97.5th percentiles of the ratio.
        (Publication Ready: Synced Visual Grammar & Terminology)
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

        # --- 统一视觉词汇表 ---
        C_VALID = '#08306B'  # Deep Navy
        C_OUTLIER = '#D55E00'  # Vermilion
        C_REPAIRED = '#27AE60'  # Green

        with cls._style_context():
            # 画布加宽以对齐其他 1x2 图表
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

            # ==========================================
            # A. Raw (Empirical Distribution)
            # ==========================================
            draw_confidence_cone(ax_raw)

            # 正常数据 (保持低调，alpha=0.5)
            ax_raw.scatter(df_raw.loc[~is_repaired, x_col], df_raw.loc[~is_repaired, y_col],
                           c=C_VALID, alpha=0.5, edgecolors='white', lw=0.5, s=60, zorder=3)
            # 异常数据 (巨型醒目橙红空心圆)
            ax_raw.scatter(df_raw.loc[is_repaired, x_col], df_raw.loc[is_repaired, y_col],
                           facecolors='none', edgecolors=C_OUTLIER, lw=2.5, s=200, alpha=0.8, zorder=4)

            ax_raw.set_title("(a) Raw (Empirical Distribution)", **SUBPLOT_TITLE_KWS)
            ax_raw.set_xlabel(x_col, fontweight='bold')
            ax_raw.set_ylabel(y_col, fontweight='bold')

            # ==========================================
            # B. Corrected (Median Aligned)
            # ==========================================
            draw_confidence_cone(ax_fix)

            # 正常数据
            ax_fix.scatter(df_fixed.loc[~is_repaired, x_col], df_fixed.loc[~is_repaired, y_col],
                           c=C_VALID, alpha=0.5, edgecolors='white', lw=0.5, s=60, zorder=3)
            # 修正后数据 (巨型醒目翠绿空心圆)
            ax_fix.scatter(df_fixed.loc[is_repaired, x_col], df_fixed.loc[is_repaired, y_col],
                           facecolors='none', edgecolors=C_REPAIRED, lw=2.5, s=200, alpha=0.8, zorder=4)

            ax_fix.set_title("(b) Corrected (Median Aligned)", **SUBPLOT_TITLE_KWS)
            ax_fix.set_xlabel(x_col, fontweight='bold')
            ax_fix.tick_params(left=False)

            sns.despine(ax=ax_raw)
            sns.despine(ax=ax_fix, left=True)

            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='#BDC3C7', lw=6, alpha=0.5, label='95% Safe Zone (2.5% - 97.5%)'),
                Line2D([0], [0], color='#34495E', ls='--', lw=2, label='Median'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C_VALID, markersize=8, alpha=0.6,
                       label='Valid Data'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_OUTLIER, markerfacecolor='none',
                       markersize=12,
                       markeredgewidth=2, label='Outliers (Raw)'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_REPAIRED, markerfacecolor='none',
                       markersize=12, markeredgewidth=2, label='Corrected Data')
            ]

            fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False,
                       fontsize=11)

            if title_on:
                fig.suptitle(main_title, y=1.02, fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.08, 1, 0.98])

            # --- 防弹保存逻辑 ---
            if is_final_record:
                safe_name = main_title.replace(":", "").replace(" ", "_").replace("(", "").replace(")",
                                                                                                   "") if main_title else "Empirical_Cone"
                cls._save_figure(fig=fig,
                                 prefix="Data Figure",
                                 experiment_name=safe_name,
                                 fitness_metric="")

            plt.show()

    @classmethod
    def plot_4c_weight_bland_altman(
            cls,
            df_raw: pd.DataFrame,
            df_fixed: pd.DataFrame,
            multi_fail_mask: pd.Series = None,
            main_title: str = None,
            is_final_record: bool = False,
            title_on: bool = True
    ):
        def get_4c_stats(df):
            temp = df.copy().reset_index(drop=True)
            # 1. 提取 4C 组成部分 (绝对质量 kg)
            water = temp['Total Body Water (TBW)']
            fat = temp['Total Fat Content (TFC)']
            protein = temp['Weight'] * (temp['Body Protein Content (Protein) (%)'] / 100.0)
            sum_3c = water + fat + protein

            # 2. 计算 Alpha (Epsilon)
            ratios = 1 - (sum_3c / temp['Weight'])
            eps_val = ratios.median()

            # 3. 基于物理约束反推观测值对应的理论体重 (Weight)
            calc_weight = sum_3c / (1 - eps_val)

            # 4. 一致性坐标 (Observed vs. 4C-Calculated)
            mean_v = (temp['Weight'] + calc_weight) / 2.0
            diff_v = temp['Weight'] - calc_weight

            return mean_v, diff_v, eps_val

        # --- 1. 数据对齐与计算 ---
        df_raw = df_raw.reset_index(drop=True)
        df_fixed = df_fixed.reset_index(drop=True)
        if multi_fail_mask is not None:
            multi_fail_mask = multi_fail_mask.reset_index(drop=True)

        mean_raw, diff_raw, eps_raw = get_4c_stats(df_raw)
        mean_fix, diff_fix, eps_fix = get_4c_stats(df_fixed)

        # --- 2. 核心逻辑：全局修复掩码 ---
        core_components = [
            'Total Body Water (TBW)',
            'Total Fat Content (TFC)',
            'Body Protein Content (Protein) (%)',
            'Muscle Mass (MM)',
            'Extracellular Water (ECW)',
            'Intracellular Water (ICW)',
            'Lean Mass (LM) (%)',
            'Total Body Fat Ratio (TBFR) (%)',
            'Visceral Muscle Area (VMA) (Kg)'
        ]

        is_repaired = pd.Series(False, index=df_raw.index)
        for col in core_components:
            if col in df_raw.columns and col in df_fixed.columns:
                changed = ~np.isclose(df_raw[col], df_fixed[col], equal_nan=True)
                is_repaired = is_repaired | changed

        mask_repaired = is_repaired
        mask_valid = ~is_repaired

        # --- 3. 视觉配置 ---
        C_VALID = '#08306B'  # 深蓝
        C_OUTLIER = '#D55E00'  # 橙红
        C_CORRECTED = '#27AE60'  # 翠绿
        C_UNFIXED = '#FF0000'

        SUBPLOT_TITLE_KWS = {
            'loc': 'left', 'fontstyle': 'italic', 'fontsize': 13, 'color': '#444444', 'pad': 15
        }

        def add_background_kde(ax, mean_series, diff_series, color):
            diff_array = diff_series.dropna().values.astype(float)
            if diff_array.std() < 1e-4:
                diff_array = diff_array + np.random.normal(0, 1e-3, len(diff_array))
            kde = stats.gaussian_kde(diff_array)
            y_vals = np.linspace(diff_array.min() - 1, diff_array.max() + 1, 300)
            density = kde(y_vals)
            x_min, x_max = mean_series.min(), mean_series.max()
            scaled_density = x_min + (density / density.max()) * ((x_max - x_min) * 0.25)
            ax.fill_betweenx(y_vals, x_min, scaled_density, color=color, alpha=0.1, zorder=0, linewidth=0)
            ax.plot(scaled_density, y_vals, color=color, alpha=0.2, zorder=0, lw=1)

        # --- 4. 绘图执行 ---
        with cls._style_context():
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True, dpi=450)

            configs = [
                {'ax': axes[0], 'mean': mean_raw, 'diff': diff_raw, 'eps': eps_raw,
                 'color': C_OUTLIER, 'fail_color': C_OUTLIER, 'title': "(a) Raw"},
                {'ax': axes[1], 'mean': mean_fix, 'diff': diff_fix, 'eps': eps_fix,
                 'color': C_CORRECTED, 'fail_color': C_UNFIXED, 'title': "(b) Corrected"}
            ]

            for cfg in configs:
                ax, m, d, eps = cfg['ax'], cfg['mean'], cfg['diff'], cfg['eps']
                add_background_kde(ax, m, d, color=cfg['color'])

                current_fail = multi_fail_mask if multi_fail_mask is not None else pd.Series(False, index=m.index)

                mask_v_clean = mask_valid & ~current_fail
                ax.scatter(m[mask_v_clean], d[mask_v_clean], c=C_VALID, alpha=0.5,
                           edgecolors='white', lw=0.5, s=60, zorder=3)

                mask_r_clean = mask_repaired & ~current_fail
                ax.scatter(m[mask_r_clean], d[mask_r_clean], facecolors='none',
                           edgecolors=cfg['color'], lw=2.5, s=200, alpha=0.8, zorder=4)

                if multi_fail_mask is not None:
                    ax.scatter(
                        m[multi_fail_mask], d[multi_fail_mask],
                        facecolors='none',
                        edgecolors=cfg['fail_color'],
                        lw=2.5, s=200, alpha=0.9,
                        zorder=5,
                        marker='o'
                    )

                avg_d, std_d = d.mean(), d.std()
                ax.axhline(avg_d, color='black', ls='-', lw=1.8, zorder=2)
                ax.axhline(avg_d + 1.96 * std_d, color='#666666', ls='--', lw=1.2, zorder=2)
                ax.axhline(avg_d - 1.96 * std_d, color='#666666', ls='--', lw=1.2, zorder=2)

                TEXT_KWS = {'transform': ax.transAxes, 'ha': 'right', 'va': 'bottom', 'fontweight': 'bold',
                            'fontsize': 11, 'zorder': 6}

                ax.text(0.98, 0.08, f'$\\alpha_{{4C}}$: {eps:.4f}', color='#333333', **TEXT_KWS)
                ax.text(0.98, 0.03, f'LoA SD: {std_d:.2f}', color='#333333', **TEXT_KWS)

                ax.set_title(cfg['title'], **SUBPLOT_TITLE_KWS)
                ax.set_xlabel("Mean Weight (Observed & 4C-Calculated) [kg]", fontweight='bold')

            axes[0].set_ylabel("Difference (Observed - 4C-Calculated) [kg]", fontweight='bold')
            sns.despine(ax=axes[0])
            sns.despine(ax=axes[1], left=True)
            axes[1].tick_params(left=False)

            # 极简统一条例
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C_VALID, markersize=8, alpha=0.6,
                       label='Valid Data'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_OUTLIER, markerfacecolor='none',
                       markersize=12, markeredgewidth=2, label='Outliers (Raw)'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_CORRECTED, markerfacecolor='none',
                       markersize=12, markeredgewidth=2, label='Corrected Data')
            ]

            if multi_fail_mask is not None:
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markeredgecolor=C_UNFIXED, markerfacecolor='none',
                           markersize=12, markeredgewidth=2, label='Uncorrectable data')
                )
                ncol_count = 4
            else:
                ncol_count = 3

            fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=ncol_count,
                       frameon=False, fontsize=11)
            if title_on:
                fig.suptitle(main_title if main_title else "4C Consistency Analysis", y=1.02, fontsize=16,
                             fontweight='bold')

            plt.tight_layout(rect=[0, 0.1, 1, 0.98])

            if is_final_record:
                safe_name = main_title.replace(":", "").replace(" ", "_").replace("(", "").replace(")",
                                                                                                   "") if main_title else "4C_Bland_Altman"
                cls._save_figure(fig=fig, prefix="Data Figure", experiment_name=safe_name, fitness_metric="")

            plt.show()

    @classmethod
    def plot_obesity_forensic_trilogy(
            cls,
            df_raw: pd.DataFrame,
            df_fixed: pd.DataFrame,
            group_col: str = 'Forensic_Group',
            main_title: str = "Forensic Dashboard: Decoding Multi-Standard Physics & Data Correction",
            is_final_record: bool = False,
            title_on: bool = True
    ):
        """
        The Forensic 2x2 Dashboard (Academic/Publication Ready).
        - Categories: Deep Blue -> Deep Purple (High Contrast Cool Tones).
        - States: Vermilion (Outlier, Hollow) / Green (Corrected, Hollow).
        - Layout: (a) Raw Global, (b) Population, (c) Raw Structural, (d) Corrected Structural.
        """
        if group_col not in df_raw.columns:
            raise ValueError(f"Missing '{group_col}'. Please run forensic logic first.")

        # --- 1. 核心学术色板 (Blue-Purple Gradient) ---
        PALETTE = {
            'Standard 22': '#084594',  # Deep Blue
            'Standard 22.5': '#4292C6',  # Mid Blue
            'Standard 23.5': '#9E9AC8',  # Light Purple Transition
            'Standard 24.5': '#6A51A3',  # Mid Purple
            'Standard 25.5': '#4A1486',  # Deep Purple
            'Unclassified': '#757575',  # Dark Gray (Grounded background)
            'Raw Outlier': '#D55E00'  # Vermilion (橙红)
        }
        C_CORRECTED = '#27AE60'  # Green (翠绿)

        # 统一子图标题样式
        SUBPLOT_TITLE_KWS = {
            'loc': 'left', 'fontstyle': 'italic', 'fontweight': 'normal',
            'fontsize': 13, 'color': '#444444', 'pad': 15
        }

        # --- 2. 统一全局视觉参数 ---
        BASE_ALPHA = 0.6  # 背景点、线统一透明度
        RING_ALPHA = 0.85  # 重点圆圈透明度
        RING_SIZE = 280  # 进一步放大的空心追踪圆 (更加醒目)
        RING_LW = 2.5  # 追踪圆线宽

        # 数据层准备
        df_normal = df_raw[df_raw[group_col] != 'Raw Outlier']
        df_outlier = df_raw[df_raw[group_col] == 'Raw Outlier']
        clean_df = df_raw[df_raw['Obesity (%)'] < 500]

        x_min, x_max = clean_df['BMI_Final'].min() * 0.9, clean_df['BMI_Final'].max() * 1.1
        y_zoom_max = clean_df['Obesity (%)'].max() * 1.15

        def plot_theoretical_v_shapes(ax, x_range):
            ax.plot(x_range, np.abs((x_range - 22.0) / 22.0) * 100, color=PALETTE['Standard 22'], ls='-',
                    alpha=BASE_ALPHA, lw=2, zorder=1)
            ax.plot(x_range, np.abs((x_range - 22.5) / 22.5) * 100, color=PALETTE['Standard 22.5'],
                    ls=(0, (3, 1, 1, 1)), alpha=BASE_ALPHA, lw=2, zorder=1)
            ax.plot(x_range, np.abs((x_range - 23.5) / 23.5) * 100, color=PALETTE['Standard 23.5'], ls='-.',
                    alpha=BASE_ALPHA, lw=2, zorder=1)
            ax.plot(x_range, np.abs((x_range - 24.5) / 24.5) * 100, color=PALETTE['Standard 24.5'], ls='--',
                    alpha=BASE_ALPHA, lw=2, zorder=1)
            ax.plot(x_range, np.abs((x_range - 25.5) / 25.5) * 100, color=PALETTE['Standard 25.5'], ls=':',
                    alpha=BASE_ALPHA, lw=2, zorder=1)

        with cls._style_context():
            fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=450, constrained_layout=True)
            x_vals = np.linspace(x_min, x_max, 300)

            # ==========================================
            # PANEL A: RAW GLOBAL (Top-Left)
            # ==========================================
            ax1 = axes[0, 0]
            sns.scatterplot(
                data=df_normal, x='BMI_Final', y='Obesity (%)',
                hue=group_col, palette=PALETTE, s=60, alpha=BASE_ALPHA, edgecolor='white', ax=ax1, legend=False,
                zorder=2
            )
            if not df_outlier.empty:
                err_x, err_y = df_outlier['BMI_Final'].iloc[0], df_outlier['Obesity (%)'].iloc[0]
                ax1.scatter(err_x, err_y, facecolors='none', edgecolors=PALETTE['Raw Outlier'],
                            marker='o', s=RING_SIZE, lw=RING_LW, alpha=RING_ALPHA, zorder=4)

            ax1.set_title("(a) Raw (Global Scale)", **SUBPLOT_TITLE_KWS)
            ax1.set_xlabel("Body Mass Index (BMI)", fontweight='bold')
            ax1.set_ylabel("Obesity (%)", fontweight='bold')

            # ==========================================
            # PANEL B: DISTRIBUTION (Top-Right)
            # ==========================================
            ax2 = axes[0, 1]
            plot_data = df_normal[group_col].value_counts()
            bar_palette = [PALETTE.get(x, '#333333') for x in plot_data.index]
            sns.barplot(x=plot_data.values, y=plot_data.index, palette=bar_palette, ax=ax2, hue=plot_data.index,
                        legend=False, alpha=0.8)

            for i, v in enumerate(plot_data.values):
                ax2.text(v + 1, i, str(v), color='black', va='center', fontweight='bold', fontsize=11)

            ax2.set_title("(b) Population Profile", **SUBPLOT_TITLE_KWS)
            ax2.set_xlabel("Patient Count", fontweight='bold')
            ax2.set_ylabel("")
            ax2.grid(axis='x', linestyle='--', alpha=0.4)

            # ==========================================
            # PANEL C: RAW STRUCTURAL (Bottom-Left)
            # ==========================================
            ax3 = axes[1, 0]
            sns.scatterplot(
                data=clean_df, x='BMI_Final', y='Obesity (%)',
                hue=group_col, palette=PALETTE, s=60, alpha=BASE_ALPHA, edgecolor='white', ax=ax3, legend=False,
                zorder=3
            )
            plot_theoretical_v_shapes(ax3, x_vals)
            if not df_outlier.empty:
                err_x, err_y = df_outlier['BMI_Final'].iloc[0], df_outlier['Obesity (%)'].iloc[0]
                ax3.scatter(err_x, err_y, facecolors='none', edgecolors=PALETTE['Raw Outlier'],
                            marker='o', s=RING_SIZE, lw=RING_LW, alpha=RING_ALPHA, zorder=4)

            ax3.set_xlim(x_min, x_max)
            ax3.set_ylim(bottom=-2, top=y_zoom_max)
            ax3.set_title("(c) Raw (Structural Physics)", **SUBPLOT_TITLE_KWS)
            ax3.set_xlabel("Body Mass Index (BMI)", fontweight='bold')
            ax3.set_ylabel("Obesity (%)", fontweight='bold')

            # ==========================================
            # PANEL D: CORRECTED STRUCTURAL (Bottom-Right)
            # ==========================================
            ax4 = axes[1, 1]
            sns.scatterplot(
                data=clean_df, x='BMI_Final', y='Obesity (%)',
                hue=group_col, palette=PALETTE, s=60, alpha=BASE_ALPHA, edgecolor='white', ax=ax4, legend=False,
                zorder=2
            )
            plot_theoretical_v_shapes(ax4, x_vals)

            if not df_outlier.empty:
                idx = df_outlier.index[0]
                fixed_x, fixed_y = df_fixed.loc[idx, 'BMI_Final'], df_fixed.loc[idx, 'Obesity (%)']
                # 翠绿空心大圆圈，精准展示 Corrected Data
                ax4.scatter(fixed_x, fixed_y, facecolors='none', edgecolors=C_CORRECTED,
                            marker='o', s=RING_SIZE, lw=RING_LW, alpha=RING_ALPHA, zorder=10)

            ax4.set_xlim(x_min, x_max)
            ax4.set_ylim(bottom=-2, top=y_zoom_max)
            ax4.set_title("(d) Corrected (Structural Physics)", **SUBPLOT_TITLE_KWS)
            ax4.set_xlabel("Body Mass Index (BMI)", fontweight='bold')
            ax4.set_ylabel("")

            # ==========================================
            # 格式化与学术图例
            # ==========================================
            for ax in [ax1, ax3, ax4]:
                sns.despine(ax=ax);
                ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
            sns.despine(ax=ax2, left=True)

            custom_lines = [
                Line2D([0], [0], color=PALETTE['Standard 22'], lw=3, label='Standard 22'),
                Line2D([0], [0], color=PALETTE['Standard 22.5'], lw=3, ls=(0, (3, 1, 1, 1)), label='Standard 22.5'),
                Line2D([0], [0], color=PALETTE['Standard 23.5'], lw=3, ls='-.', label='Standard 23.5'),
                Line2D([0], [0], color=PALETTE['Standard 24.5'], lw=3, ls='--', label='Standard 24.5'),
                Line2D([0], [0], color=PALETTE['Standard 25.5'], lw=3, ls=':', label='Standard 25.5'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=PALETTE['Raw Outlier'], markerfacecolor='none',
                       markersize=14, markeredgewidth=2.5, label='Outliers (Raw)'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor=C_CORRECTED, markerfacecolor='none',
                       markersize=14, markeredgewidth=2.5, label='Corrected Data')
            ]
            fig.legend(handles=custom_lines, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=7, frameon=False,
                       fontsize=12)
            if title_on:
                fig.suptitle(main_title, y=1.04, fontsize=18,
                         fontweight='bold')

            if is_final_record:
                cls._save_figure(fig=fig,
                                 prefix="Data Figure",
                                 experiment_name=main_title,
                                 fitness_metric="")

            plt.show()