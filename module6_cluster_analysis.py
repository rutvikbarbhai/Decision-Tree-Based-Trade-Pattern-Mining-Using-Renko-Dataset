#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze clusters and use them for trade classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib

sns.set_style('whitegrid')


class ClusterAnalyzer:
    """Analyze and interpret clustering results"""

    def __init__(self, model, df, features, target_col='label', dataset_name='dataset'):
        self.model = model
        self.df = df
        self.features = features
        self.target_col = target_col
        self.dataset_name = dataset_name

        # Get cluster assignments
        self.df['cluster'] = model.predict(df[features].fillna(0))
        self.df['leaf_id'] = model.apply(df[features].fillna(0))

    def analyze_cluster_characteristics(self):
        """Analyze what makes each cluster unique"""

        print("=" * 70)
        print(f"CLUSTER CHARACTERISTICS ANALYSIS - {self.dataset_name.upper()}")
        print("=" * 70)
        print(f"\nAnalyzing {self.df['leaf_id'].nunique()} decision tree leaves (clusters)")

        cluster_stats = []

        for cluster in sorted(self.df['leaf_id'].unique()):
            cluster_df = self.df[self.df['leaf_id'] == cluster]

            stats = {
                'cluster': int(cluster),
                'size': int(len(cluster_df)),
                'size_pct': float(len(cluster_df) / len(self.df) * 100),
                'target_distribution': {str(k): int(v) for k, v in
                                        cluster_df[self.target_col].value_counts().to_dict().items()},
                'purity': float(cluster_df[self.target_col].value_counts().max() / len(cluster_df))
            }

            # Get mean feature values for this cluster
            feature_means = cluster_df[self.features].mean()

            # Find distinguishing features (compare to global mean)
            global_means = self.df[self.features].mean()

            # Calculate relative differences
            differences = pd.Series(index=feature_means.index, dtype=float)
            for feat in feature_means.index:
                if abs(global_means[feat]) < 1e-10:
                    differences[feat] = abs(feature_means[feat])
                else:
                    differences[feat] = abs((feature_means[feat] - global_means[feat]) / global_means[feat])

            top_diff_features = differences.nlargest(5)

            stats['distinguishing_features'] = {
                feat: {
                    'cluster_mean': float(feature_means[feat]),
                    'global_mean': float(global_means[feat]),
                    'diff_pct': float(differences[feat] * 100)
                }
                for feat in top_diff_features.index
            }

            cluster_stats.append(stats)

        # Print summary for first 10 clusters
        print(f"\nShowing first 10 clusters (of {len(cluster_stats)} total):")
        for stat in cluster_stats[:10]:
            print(f"\n{'=' * 70}")
            print(f"CLUSTER (Leaf ID): {stat['cluster']}")
            print(f"{'=' * 70}")
            print(f"Size: {stat['size']:,} samples ({stat['size_pct']:.1f}%)")
            print(f"Purity: {stat['purity']:.2%}")
            print("\nTarget Distribution:")
            for target, count in stat['target_distribution'].items():
                pct = count / stat['size'] * 100
                print(f"  {target}: {count:,} ({pct:.1f}%)")

            print("\nTop 5 Distinguishing Features:")
            for feat, values in list(stat['distinguishing_features'].items())[:5]:
                print(f"  {feat}:")
                print(f"    Cluster mean: {values['cluster_mean']:.3f}")
                print(f"    Global mean: {values['global_mean']:.3f}")
                print(f"    Difference: {values['diff_pct']:.1f}%")

        return cluster_stats

    # def identify_actionable_clusters(self, min_purity=0.7, min_size=50):
    #     """Find clusters with high purity (good for trading signals)"""
    #
    #     print("=" * 70)
    #     print(f"ACTIONABLE CLUSTERS - {self.dataset_name.upper()}")
    #     print("=" * 70)
    #     print(f"\nCriteria: Purity >= {min_purity:.0%}, Size >= {min_size}")
    #
    #     actionable = []
    #
    #     for cluster in sorted(self.df['leaf_id'].unique()):
    #         cluster_df = self.df[self.df['leaf_id'] == cluster]
    #         size = len(cluster_df)
    #         purity = cluster_df[self.target_col].value_counts().max() / size
    #
    #         if purity >= min_purity and size >= min_size:
    #             predicted_class = cluster_df[self.target_col].value_counts().idxmax()
    #
    #             actionable.append({
    #                 'cluster': int(cluster),
    #                 'size': int(size),
    #                 'purity': float(purity),
    #                 'predicted_class': int(predicted_class),
    #                 'confidence': float(purity),
    #                 'signal_strength': float(purity * np.log(size))
    #             })
    #
    #     # Sort by signal strength
    #     actionable = sorted(actionable, key=lambda x: x['signal_strength'], reverse=True)
    #
    #     print(f"\nFound {len(actionable)} actionable clusters:")
    #     print(f"\n{'Cluster':<10} {'Size':<8} {'Purity':<10} {'Class':<8} {'Signal':<10}")
    #     print("-" * 70)
    #
    #     for cluster in actionable[:15]:
    #         print(f"{cluster['cluster']:<10} "
    #               f"{cluster['size']:<8} "
    #               f"{cluster['purity']:<10.2%} "
    #               f"{cluster['predicted_class']:<8} "
    #               f"{cluster['signal_strength']:<10.2f}")
    #
    #     return actionable

    def identify_actionable_clusters(self, min_purity=0.7, min_size=50, balance=True, max_clusters_per_class=30):
        """Find clusters with optional class balancing"""

        actionable = []

        for cluster in sorted(self.df['leaf_id'].unique()):
            cluster_df = self.df[self.df['leaf_id'] == cluster]
            size = len(cluster_df)
            purity = cluster_df[self.target_col].value_counts().max() / size

            if purity >= min_purity and size >= min_size:
                predicted_class = cluster_df[self.target_col].value_counts().idxmax()

                actionable.append({
                    'cluster': int(cluster),
                    'size': int(size),
                    'purity': float(purity),
                    'predicted_class': int(predicted_class),
                    'confidence': float(purity),
                    'signal_strength': float(purity * np.log(size))
                })

        if balance:
            win_clusters = sorted([c for c in actionable if c['predicted_class'] == 1],
                                  key=lambda x: x['signal_strength'], reverse=True)
            loss_clusters = sorted([c for c in actionable if c['predicted_class'] == 0],
                                   key=lambda x: x['signal_strength'], reverse=True)

            # Take equal number from each class (or all WIN clusters if fewer)
            n_per_class = min(len(win_clusters), len(loss_clusters), max_clusters_per_class)

            actionable = win_clusters[:n_per_class] + loss_clusters[:n_per_class]

            print(f"\n BALANCED: {n_per_class} WIN + {n_per_class} LOSS clusters")

        actionable = sorted(actionable, key=lambda x: x['signal_strength'], reverse=True)

        return actionable

    def plot_cluster_distribution(self):
        """Plot cluster size distribution"""

        cluster_sizes = self.df['leaf_id'].value_counts().sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        ax1 = axes[0]
        cluster_sizes.head(20).plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Cluster (Leaf ID)', fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontweight='bold')
        ax1.set_title(f'Top 20 Cluster Sizes - {self.dataset_name}', fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3)

        # Histogram
        ax2 = axes[1]
        ax2.hist(cluster_sizes.values, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Cluster Size', fontweight='bold')
        ax2.set_ylabel('Number of Clusters', fontweight='bold')
        ax2.set_title(f'Cluster Size Distribution - {self.dataset_name}', fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def classify_new_trade(self, trade_features):
        """Classify a new trade using the decision tree"""

        # Convert to DataFrame if needed
        if isinstance(trade_features, pd.Series):
            trade_features = trade_features.to_frame().T

        # Get prediction and cluster assignment
        prediction = self.model.predict(trade_features.fillna(0))[0]
        leaf_id = self.model.apply(trade_features.fillna(0))[0]
        proba = self.model.predict_proba(trade_features.fillna(0))[0]

        # Get cluster statistics
        cluster_trades = self.df[self.df['leaf_id'] == leaf_id]
        confidence = cluster_trades[self.target_col].value_counts().max() / len(cluster_trades)

        print(f"\nTrade Classification ({self.dataset_name}):")
        print(f"  Predicted Class: {prediction}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Cluster (Leaf ID): {leaf_id}")
        print(f"  Size: {len(cluster_trades):,} similar trades")
        print("  Target distribution:")
        for target, count in cluster_trades[self.target_col].value_counts().items():
            pct = count / len(cluster_trades) * 100
            print(f"    {target}: {count} ({pct:.1f}%)")

        return {
            'prediction': prediction,
            'confidence': confidence,
            'leaf_id': leaf_id,
            'cluster_size': len(cluster_trades),
            'probabilities': dict(zip(self.model.classes_, proba))
        }


def compare_train_test_clusters(train_stats, test_stats):
    """Compare cluster performance between train and test"""

    print("\n" + "=" * 70)
    print("TRAIN vs TEST CLUSTER COMPARISON")
    print("=" * 70)

    # Find clusters that appear in both
    train_clusters = {s['cluster'] for s in train_stats}
    test_clusters = {s['cluster'] for s in test_stats}
    common_clusters = train_clusters & test_clusters

    print("\nCluster Coverage:")
    print(f"  Train only: {len(train_clusters - test_clusters)} clusters")
    print(f"  Test only: {len(test_clusters - train_clusters)} clusters")
    print(f"  Common: {len(common_clusters)} clusters")

    # Compare purity for common clusters
    print("\nPurity Comparison (Common Clusters):")
    print(f"{'Cluster':<10} {'Train Purity':<15} {'Test Purity':<15} {'Difference':<12}")
    print("-" * 70)

    comparisons = []
    for cluster_id in sorted(list(common_clusters))[:15]:
        train_stat = next((s for s in train_stats if s['cluster'] == cluster_id), None)
        test_stat = next((s for s in test_stats if s['cluster'] == cluster_id), None)

        if train_stat and test_stat:
            train_purity = train_stat['purity']
            test_purity = test_stat['purity']
            diff = test_purity - train_purity

            comparisons.append({
                'cluster': cluster_id,
                'train_purity': train_purity,
                'test_purity': test_purity,
                'difference': diff
            })

            print(f"{cluster_id:<10} {train_purity:<15.2%} {test_purity:<15.2%} {diff:>+11.2%}")

    # Summary statistics
    if comparisons:
        avg_train_purity = np.mean([c['train_purity'] for c in comparisons])
        avg_test_purity = np.mean([c['test_purity'] for c in comparisons])
        avg_diff = np.mean([c['difference'] for c in comparisons])

        print(f"\n{'Average:':<10} {avg_train_purity:<15.2%} {avg_test_purity:<15.2%} {avg_diff:>+11.2%}")

        print("\n Cluster Stability Assessment:")
        if abs(avg_diff) < 0.05:
            print("  EXCELLENT: Clusters are very stable (<5% difference)")
        elif abs(avg_diff) < 0.10:
            print("  GOOD: Clusters are reasonably stable (<10% difference)")
        elif abs(avg_diff) < 0.15:
            print("  FAIR: Some cluster degradation (10-15% difference)")
        else:
            print("  WARNING: Significant cluster degradation (>15% difference)")
            print("   Clusters may not generalize well to new data")

    return comparisons


def run_cluster_analysis(train_path, test_path, model_dir, features_path, output_dir, type):
    """Run cluster analysis workflow for one dataset (long or short)."""

    name = type.lower().replace("_trades", "")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print(f"RUNNING CLUSTER ANALYSIS FOR: {output_dir}")
    print("=" * 80)

    # === Load data & model ===
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    model = joblib.load(Path(model_dir) / f'decision_tree_model_{name}.pkl')

    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]

    print(f"Loaded {len(train_df):,} train, {len(test_df):,} test samples")

    analyzer_train = ClusterAnalyzer(model, train_df, features, target_col='label', dataset_name='TRAIN')
    analyzer_test = ClusterAnalyzer(model, test_df, features, target_col='label', dataset_name='TEST')

    # === Core steps ===
    print("\nAnalyzing TRAIN clusters...")
    train_stats = analyzer_train.analyze_cluster_characteristics()

    if type == 'long_trades':
        train_purity = 0.65
        train_min_samples = 15
        test_purity = 0.65
        test_min_samples = 10

    else:
        train_purity = 0.65
        train_min_samples = 15
        test_purity = 0.65
        test_min_samples = 10

    train_actionable = analyzer_train.identify_actionable_clusters(
        min_purity=train_purity,
        min_size=train_min_samples,
        balance=True,
        max_clusters_per_class=20
    )

    print("\nAnalyzing TEST clusters...")
    test_stats = analyzer_test.analyze_cluster_characteristics()

    # test_actionable = analyzer_test.identify_actionable_clusters(min_purity=purity_threshold, min_size=20)
    # test_actionable = analyzer_test.identify_actionable_clusters(min_purity=0.6, min_size=50)

    test_actionable = analyzer_test.identify_actionable_clusters(
        min_purity=test_purity,
        min_size=test_min_samples,
        balance=False,
        max_clusters_per_class=20
    )

    print("\nComparing TRAIN vs TEST clusters...")
    comparisons = compare_train_test_clusters(train_stats, test_stats)

    # === Save outputs ===
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"cluster_statistics_train_{name}.json", "w") as f:
        json.dump(train_stats, f, indent=2)
    with open(output_dir / f"cluster_statistics_test_{name}.json", "w") as f:
        json.dump(test_stats, f, indent=2)
    with open(output_dir / f"actionable_clusters_train_{name}.json", "w") as f:
        json.dump(train_actionable, f, indent=2)
    with open(output_dir / f"actionable_clusters_test_{name}.json", "w") as f:
        json.dump(test_actionable, f, indent=2)
    with open(output_dir / f"train_test_comparison_{name}.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    print(f"\n Completed: {output_dir}")
    return analyzer_train, analyzer_test, comparisons


def main():
    """Run cluster analysis for both LONG and SHORT trades."""
    test_name = "base_ind_osc"
    configs = [
        {
            "name": "Long Trades",
            "train": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/train_prepared_long.parquet",
            "test": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/test_prepared_long.parquet",
            "features": f"./outputs/test_{test_name}/long_trades/module4_feature_reduction_long_trades/final_features_long.txt",
            "model_dir": f"./outputs/test_{test_name}/long_trades/module5_supervised_clustering_long_trades",
            "output": f"./outputs/test_{test_name}/long_trades/module6_cluster_analysis_long_trades"
        },
        {
            "name": "Short Trades",
            "train": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/train_prepared_short.parquet",
            "test": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/test_prepared_short.parquet",
            "features": f"./outputs/test_{test_name}/short_trades/module4_feature_reduction_short_trades/final_features_short.txt",
            "model_dir": f"./outputs/test_{test_name}/short_trades/module5_supervised_clustering_short_trades",
            "output": f"./outputs/test_{test_name}/short_trades/module6_cluster_analysis_short_trades"
        }
    ]

    all_results = {}
    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"RUNNING {cfg['name'].upper()} ANALYSIS")
        print("=" * 80)

        type = cfg['name'].lower().replace(' ', '_')

        analyzer_train, analyzer_test, comparison = run_cluster_analysis(
            cfg["train"], cfg["test"], cfg["model_dir"], cfg["features"], cfg["output"], type
        )

        all_results[cfg["name"]] = {
            "train": analyzer_train,
            "test": analyzer_test,
            "comparison": comparison
        }

    print("\n" + "=" * 80)
    print("ALL CLUSTER ANALYSES COMPLETED")
    print("=" * 80)
    return all_results


if __name__ == "__main__":
    results_summary = main()
