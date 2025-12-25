#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 11: Walk-Forward Analysis

More rigorous backtesting approach:
1. Splits data into multiple time windows
2. Tests pattern stability across different periods
3. Detects when patterns stop working
4. Provides rolling performance metrics
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


class WalkForwardAnalyzer:
    """Perform walk-forward analysis on trading rules"""

    def __init__(self, model, features, rule_catalog, df, name):
        self.model = model
        self.features = features
        self.rule_catalog = rule_catalog
        self.name = name
        self.df = df.copy()

        # Add predictions
        self.df['predicted_cluster'] = model.apply(df[features].fillna(0))
        self.df['predicted_label'] = model.predict(df[features].fillna(0))

    def create_time_folds(self, n_folds=5):
        """Create sequential time-based folds"""

        print("=" * 80)
        print(f"CREATING {n_folds} TIME-BASED FOLDS")
        print("=" * 80)

        # Sort by date if available
        if 'entry_date' in self.df.columns:
            self.df = self.df.sort_values('entry_date')
            has_dates = True
        else:
            has_dates = False

        fold_size = len(self.df) // n_folds
        folds = []

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(self.df)

            fold_df = self.df.iloc[start_idx:end_idx].copy()

            fold_info = {
                'fold_number': i + 1,
                'data': fold_df,
                'size': len(fold_df),
                'start_idx': start_idx,
                'end_idx': end_idx
            }

            if has_dates:
                fold_info['start_date'] = fold_df['entry_date'].min()
                fold_info['end_date'] = fold_df['entry_date'].max()
                print(f"\nFold {i + 1}: {fold_info['start_date']} to {fold_info['end_date']} "
                      f"({fold_info['size']:,} samples)")
            else:
                print(f"\nFold {i + 1}: Rows {start_idx:,} to {end_idx:,} "
                      f"({fold_info['size']:,} samples)")

            folds.append(fold_info)

        return folds

    def evaluate_cluster_by_fold(self, folds, cluster_id):
        """Track a cluster's performance across all folds"""

        fold_results = []

        for fold in folds:
            fold_df = fold['data']

            # Get samples in this cluster
            cluster_mask = fold_df['predicted_cluster'] == cluster_id
            cluster_samples = fold_df[cluster_mask]

            if len(cluster_samples) > 0:
                actual_labels = cluster_samples['label']

                # Get expected label from rule catalog
                cluster_rules = [r for r in self.rule_catalog['all_rules']
                                 if r['cluster_id'] == cluster_id]
                expected_label = cluster_rules[0]['predicted_class'] if cluster_rules else None

                correct = (actual_labels == expected_label).sum()
                accuracy = correct / len(actual_labels)

                result = {
                    'fold': fold['fold_number'],
                    'samples': len(actual_labels),
                    'accuracy': accuracy,
                    'correct': correct
                }

                if 'start_date' in fold:
                    result['date'] = fold['start_date']

                fold_results.append(result)

        return fold_results

    def analyze_all_clusters(self, folds):
        """Analyze all actionable clusters across folds"""

        print("\n" + "=" * 80)
        print("WALK-FORWARD ANALYSIS: ALL CLUSTERS")
        print("=" * 80)

        all_results = {}
        actionable_ids = [rule['cluster_id'] for rule in self.rule_catalog['all_rules']]

        for cluster_id in actionable_ids:
            fold_results = self.evaluate_cluster_by_fold(folds, cluster_id)

            if len(fold_results) >= 3:  # Need at least 3 folds with data
                all_results[cluster_id] = fold_results

        print(f"\nAnalyzed {len(all_results)} clusters across {len(folds)} time periods")

        return all_results

    def calculate_stability_metrics(self, cluster_results):
        """Calculate stability metrics for each cluster"""

        print("\n" + "=" * 80)
        print("CALCULATING STABILITY METRICS")
        print("=" * 80)

        stability_data = []

        for cluster_id, fold_results in cluster_results.items():
            accuracies = [r['accuracy'] for r in fold_results]

            stability_metrics = {
                'cluster_id': cluster_id,
                'n_periods': len(fold_results),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'range': np.max(accuracies) - np.min(accuracies),
                'coefficient_of_variation': np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0
            }

            # Get original train confidence
            cluster_rules = [r for r in self.rule_catalog['all_rules']
                             if r['cluster_id'] == cluster_id]
            if cluster_rules:
                stability_metrics['train_confidence'] = cluster_rules[0]['confidence']
                stability_metrics['degradation'] = cluster_rules[0]['confidence'] - stability_metrics['mean_accuracy']

            stability_data.append(stability_metrics)

        stability_df = pd.DataFrame(stability_data)
        stability_df = stability_df.sort_values('mean_accuracy', ascending=False)

        # Identify most stable patterns
        # Low coefficient of variation = more stable
        stable_patterns = stability_df[
            (stability_df['coefficient_of_variation'] < 0.15)
            & (stability_df['mean_accuracy'] > 0.7)
        ]

        print(f"\nFound {len(stable_patterns)} highly stable patterns:")
        print("-" * 80)

        for idx, row in stable_patterns.head(10).iterrows():
            print(f"Cluster {int(row['cluster_id']):3d} | "
                  f"Mean: {row['mean_accuracy']:.2%} | "
                  f"Std: {row['std_accuracy']:.3f} | "
                  f"Range: {row['range']:.2%} | "
                  f"CV: {row['coefficient_of_variation']:.3f}")

        return stability_df

    def identify_degrading_patterns(self, cluster_results):
        """Identify patterns that degrade over time"""

        print("\n" + "=" * 80)
        print("IDENTIFYING DEGRADING PATTERNS")
        print("=" * 80)

        degrading = []

        for cluster_id, fold_results in cluster_results.items():
            if len(fold_results) < 3:
                continue

            accuracies = [r['accuracy'] for r in fold_results]

            # Check if there's a downward trend
            # Simple approach: compare first half vs second half
            mid = len(accuracies) // 2
            first_half_avg = np.mean(accuracies[:mid])
            second_half_avg = np.mean(accuracies[mid:])

            drop = first_half_avg - second_half_avg

            if drop > 0.1:  # 10% drop
                degrading.append({
                    'cluster_id': cluster_id,
                    'first_half_avg': first_half_avg,
                    'second_half_avg': second_half_avg,
                    'drop': drop,
                    'drop_pct': (drop / first_half_avg * 100) if first_half_avg > 0 else 0
                })

        if len(degrading) > 0:
            degrading_df = pd.DataFrame(degrading).sort_values('drop', ascending=False)

            print(f"\nFound {len(degrading_df)} patterns with significant degradation:")
            print("-" * 80)

            for idx, row in degrading_df.head(10).iterrows():
                print(f"Cluster {int(row['cluster_id']):3d} | "
                      f"Early: {row['first_half_avg']:.2%} | "
                      f"Late: {row['second_half_avg']:.2%} | "
                      f"Drop: {row['drop']:.2%} ({row['drop_pct']:.1f}%)")

            return degrading_df
        else:
            print("\nNo significantly degrading patterns detected")
            return pd.DataFrame()

    def create_stability_heatmap(self, cluster_results, output_dir):
        """Create heatmap showing cluster performance over time"""

        print("\n" + "=" * 80)
        print("CREATING STABILITY HEATMAP")
        print("=" * 80)

        # Prepare data for heatmap
        cluster_ids = sorted(cluster_results.keys())[:20]  # Top 20 clusters

        if len(cluster_ids) == 0:
            print("Not enough data for heatmap")
            return None

        # Get max number of folds
        max_folds = max(len(results) for results in cluster_results.values())

        # Create matrix
        heatmap_data = []
        labels = []

        for cluster_id in cluster_ids:
            fold_results = cluster_results[cluster_id]
            accuracies = [r['accuracy'] for r in fold_results]

            # Pad if needed
            while len(accuracies) < max_folds:
                accuracies.append(np.nan)

            heatmap_data.append(accuracies)
            labels.append(f"C{cluster_id}")

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

        # Set ticks
        ax.set_xticks(np.arange(max_folds))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels([f"Period {i + 1}" for i in range(max_folds)])
        ax.set_yticklabels(labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy', rotation=270, labelpad=20, fontweight='bold')

        # Add text annotations
        for i in range(len(labels)):
            for j in range(max_folds):
                if not np.isnan(heatmap_data[i][j]):
                    ax.text(j, i, f'{heatmap_data[i][j]:.2f}',
                            ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Cluster Performance Over Time', fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Time Period', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cluster ID', fontweight='bold', fontsize=12)

        plt.tight_layout()

        save_path = Path(output_dir) / f'stability_heatmap_{self.name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\nHeatmap saved: {save_path}")

        return fig

    def create_performance_trends(self, cluster_results, output_dir, top_n=8):
        """Create line plots showing performance trends"""

        print("\n" + "=" * 80)
        print("CREATING PERFORMANCE TREND CHARTS")
        print("=" * 80)

        # Get top clusters by sample size
        cluster_sizes = {
            cid: sum(r['samples'] for r in results)
            for cid, results in cluster_results.items()
        }
        top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, (cluster_id, _) in enumerate(top_clusters):
            ax = axes[idx]

            fold_results = cluster_results[cluster_id]
            periods = [r['fold'] for r in fold_results]
            accuracies = [r['accuracy'] for r in fold_results]

            ax.plot(periods, accuracies, marker='o', linewidth=2, markersize=8,
                    color='steelblue', label=f'Cluster {cluster_id}')
            ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(0.85, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(0.95, color='green', linestyle='--', alpha=0.5, linewidth=1)

            ax.set_xlabel('Time Period', fontweight='bold')
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.set_title(f'Cluster {cluster_id}', fontweight='bold', fontsize=12)
            ax.set_ylim(0.4, 1.05)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = Path(output_dir) / f'performance_trends_{self.name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\nTrend charts saved: {save_path}")

        return fig

    def generate_walkforward_report(self, stability_df, degrading_df, output_dir):
        """Generate comprehensive walk-forward report"""

        print("\n" + "=" * 80)
        print("GENERATING WALK-FORWARD REPORT")
        print("=" * 80)

        output_dir = Path(output_dir)

        # Save stability metrics
        stability_df.to_csv(output_dir / f'walkforward_stability_{self.name}.csv',
                            index=False, encoding='utf-8')

        if len(degrading_df) > 0:
            degrading_df.to_csv(output_dir / f'walkforward_degrading_{self.name}.csv',
                                index=False, encoding='utf-8')

        # Calculate overall accuracy
        overall_accuracy = (self.df['predicted_label'] == self.df['label']).mean()

        # Create text summary
        with open(output_dir / f'walkforward_summary_{self.name}.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("WALK-FORWARD ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Overall Model Accuracy: {overall_accuracy:.2%}\n")
            f.write(f"Total Samples: {len(self.df):,}\n\n")

            f.write("STABILITY ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Clusters Analyzed: {len(stability_df)}\n")
            f.write(
                f"Highly Stable (CV<0.15 & Acc>70%): {len(stability_df[(stability_df['coefficient_of_variation'] < 0.15) & (stability_df['mean_accuracy'] > 0.7)])}\n")
            f.write(f"Average Cluster Accuracy: {stability_df['mean_accuracy'].mean():.2%}\n")
            f.write(f"Average Std Dev: {stability_df['std_accuracy'].mean():.3f}\n\n")

            if len(degrading_df) > 0:
                f.write("DEGRADATION ANALYSIS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Degrading Patterns: {len(degrading_df)}\n")
                f.write(f"Average Drop: {degrading_df['drop'].mean():.2%}\n\n")

            f.write("TOP 10 MOST STABLE PATTERNS:\n")
            f.write("-" * 80 + "\n")
            for idx, row in stability_df.head(10).iterrows():
                f.write(f"Cluster {int(row['cluster_id']):3d}: "
                        f"Mean={row['mean_accuracy']:.2%}, "
                        f"Std={row['std_accuracy']:.3f}, "
                        f"CV={row['coefficient_of_variation']:.3f}\n")

        print(f"\nWalk-forward analysis saved to: {output_dir}")
        print(f"  - walkforward_stability_{self.name}.csv")
        if len(degrading_df) > 0:
            print(f"  - walkforward_degrading_{self.name}.csv")
        print(f"  - walkforward_summary_{self.name}.txt")
        print(f"  - stability_heatmap_{self.name}_{self.name}.png")
        print(f"  - performance_trends_{self.name}_{self.name}.png")


def run_walkforward(train_path, test_path, model_path, features_path, rule_catalog_path, output_dir, type):
    """Run full walk-forward analysis for one dataset (long or short)."""

    name = type.replace("_trades", "")
    print("=" * 80)
    print(f"RUNNING WALK-FORWARD ANALYSIS FOR: {output_dir}")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Load data and model ===
    print("\n1. Loading data and model...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_df['split'] = 'train'
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df], ignore_index=True)

    model = joblib.load(model_path)

    with open(features_path, 'r', encoding='utf-8') as f:
        features = [line.strip() for line in f.readlines()]

    with open(rule_catalog_path, 'r', encoding='utf-8') as f:
        rule_catalog = json.load(f)

    print(f"    Loaded {len(df):,} samples")
    print(f"    Loaded {len(features)} features")

    # === Create analyzer ===
    analyzer = WalkForwardAnalyzer(model, features, rule_catalog, df, name)

    # === Walk-forward steps ===
    folds = analyzer.create_time_folds(n_folds=5)
    cluster_results = analyzer.analyze_all_clusters(folds)
    stability_df = analyzer.calculate_stability_metrics(cluster_results)
    degrading_df = analyzer.identify_degrading_patterns(cluster_results)

    analyzer.create_stability_heatmap(cluster_results, output_dir)
    analyzer.create_performance_trends(cluster_results, output_dir)
    analyzer.generate_walkforward_report(stability_df, degrading_df, output_dir)
    predictions = analyzer.df
    col = predictions['predicted_label']
    predictions = predictions.drop(columns='predicted_label')
    predictions.insert(11, 'predicted_label', col)
    predictions.to_csv(f'{output_dir}/predictions_{type}.csv', index=False)

    print(f"\n Walk-forward analysis complete for: {output_dir}")
    return analyzer, stability_df, degrading_df


def main():
    """Run walk-forward analysis for both LONG and SHORT datasets."""

    test_name = "base_ind_osc"

    configs = [
        {
            "name": "Long Trades",
            "train": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/train_prepared_long.parquet",
            "test": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/test_prepared_long.parquet",
            "features": f"./outputs/test_{test_name}/long_trades/module4_feature_reduction_long_trades/final_features_long.txt",
            "model": f"./outputs/test_{test_name}/long_trades/module5_supervised_clustering_long_trades/decision_tree_model_long.pkl",
            "rules": f"./outputs/test_{test_name}/long_trades/module7_rule_extraction_long_trades/rule_catalog_long.json",
            "output": f"./outputs/test_{test_name}/long_trades/module10_walkforward"
        },
        {
            "name": "Short Trades",
            "train": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/train_prepared_short.parquet",
            "test": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/test_prepared_short.parquet",
            "features": f"./outputs/test_{test_name}/short_trades/module4_feature_reduction_short_trades/final_features_short.txt",
            "model": f"./outputs/test_{test_name}/short_trades/module5_supervised_clustering_short_trades/decision_tree_model_short.pkl",
            "rules": f"./outputs/test_{test_name}/short_trades/module7_rule_extraction_short_trades/rule_catalog_short.json",
            "output": f"./outputs/test_{test_name}/short_trades/module10_walkforward"
        }
    ]

    all_results = {}

    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"STARTING WALK-FORWARD ANALYSIS FOR {cfg['name'].upper()}")
        print("=" * 80)

        type = cfg['name'].lower().replace(" ", "_")

        try:
            analyzer, stability_df, degrading_df = run_walkforward(
                cfg["train"], cfg["test"], cfg["model"],
                cfg["features"], cfg["rules"], cfg["output"], type
            )
            all_results[cfg["name"]] = {
                "analyzer": analyzer,
                "stability_df": stability_df,
                "degrading_df": degrading_df
            }
        except Exception as e:
            print(f"ERROR during {cfg['name']} analysis: {e}")

    print("\n" + "=" * 80)
    print("ALL WALK-FORWARD ANALYSES COMPLETE ")
    print("=" * 80)

    for name, result in all_results.items():
        stable = len(result["stability_df"][
            (result["stability_df"]['coefficient_of_variation'] < 0.15)
            & (result["stability_df"]['mean_accuracy'] > 0.70)
        ])
        print(f"\n{name}:")
        print(f"  Clusters analyzed: {len(result['stability_df'])}")
        print(f"  Stable patterns:   {stable}")
        print(f"  Degrading patterns:{len(result['degrading_df'])}")

    return all_results


if __name__ == "__main__":
    results_summary = main()
