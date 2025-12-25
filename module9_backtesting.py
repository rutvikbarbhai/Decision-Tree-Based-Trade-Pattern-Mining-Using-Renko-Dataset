#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 10: Backtesting & Validation Framework

This module:
1. Tests extracted rules on out-of-sample data
2. Validates temporal stability (time-based splits)
3. Tracks performance by cluster and tier
4. Calculates risk-adjusted metrics
5. Identifies pattern degradation
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sns.set_style('whitegrid')


class BacktestEngine:
    """Backtest trading rules on historical data"""

    def __init__(self, model, features, rule_catalog, df, name):
        """
        Args:
            model: Trained DecisionTreeClassifier
            features: List of feature names
            rule_catalog: Dict from module4 with rules
            df: DataFrame with features and labels
        """
        self.model = model
        self.features = features
        self.rule_catalog = rule_catalog
        self.name = name
        self.df = df.copy()

        # Add predictions
        self.df['predicted_cluster'] = model.apply(df[features].fillna(0))
        self.df['predicted_label'] = model.predict(df[features].fillna(0))

        # Create cluster lookup
        self.cluster_info = {
            int(rule['cluster_id']): rule
            for rule in rule_catalog['all_rules']
        }

    def time_based_split(self, test_size=0.2):
        """Split data by time (not random)"""

        print("=" * 80)
        print("TIME-BASED DATA SPLIT")
        print("=" * 80)

        # Ensure data is sorted by time
        if 'entry_date' in self.df.columns:
            self.df = self.df.sort_values('entry_date')
            split_idx = int(len(self.df) * (1 - test_size))

            train_df = self.df.iloc[:split_idx].copy()
            test_df = self.df.iloc[split_idx:].copy()

            print(f"\nTrain Period: {train_df['entry_date'].min()} to {train_df['entry_date'].max()}")
            print(f"Test Period:  {test_df['entry_date'].min()} to {test_df['entry_date'].max()}")
        else:
            # Fallback to simple split if no date column
            split_idx = int(len(self.df) * (1 - test_size))
            train_df = self.df.iloc[:split_idx].copy()
            test_df = self.df.iloc[split_idx:].copy()

            print("\nWarning: No 'entry_date' column found. Using sequential split.")
            print(f"Train samples: First {len(train_df):,} rows")
            print(f"Test samples:  Last {len(test_df):,} rows")

        print(f"\nTrain set: {len(train_df):,} samples ({len(train_df) / len(self.df) * 100:.1f}%)")
        print(f"Test set:  {len(test_df):,} samples ({len(test_df) / len(self.df) * 100:.1f}%)")

        return train_df, test_df

    def evaluate_overall_performance(self, test_df):
        """Evaluate overall model performance"""

        print("\n" + "=" * 80)
        print("OVERALL PERFORMANCE ON TEST SET")
        print("=" * 80)

        y_true = test_df['label']
        y_pred = test_df['predicted_label']

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        print("\nTest Set Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

        # Class distribution
        print("\nActual Distribution:")
        print(y_true.value_counts(normalize=True).to_string())

        print("\nPredicted Distribution:")
        print(y_pred.value_counts(normalize=True).to_string())

        return metrics

    def evaluate_by_cluster(self, test_df):
        """Evaluate performance for each actionable cluster"""

        print("\n" + "=" * 80)
        print("PERFORMANCE BY CLUSTER (ACTIONABLE ONLY)")
        print("=" * 80)

        cluster_results = []

        # Get actionable cluster IDs
        actionable_ids = set(int(rule['cluster_id']) for rule in self.rule_catalog['all_rules'])

        for cluster_id in sorted(actionable_ids):
            # Get samples in this cluster
            cluster_mask = test_df['predicted_cluster'] == cluster_id
            cluster_samples = test_df[cluster_mask]

            if len(cluster_samples) == 0:
                continue

            # Get cluster info
            cluster_info = self.cluster_info.get(cluster_id, {})
            expected_label = cluster_info.get('predicted_class', None)
            train_confidence = cluster_info.get('confidence', 0)

            # Calculate test performance
            actual_labels = cluster_samples['label']
            correct = (actual_labels == expected_label).sum()
            total = len(actual_labels)
            test_accuracy = correct / total if total > 0 else 0

            # Performance degradation
            degradation = train_confidence - test_accuracy

            result = {
                'cluster_id': int(cluster_id),
                'expected_label': int(expected_label) if expected_label is not None else None,
                'train_confidence': float(train_confidence),
                'test_samples': int(total),
                'test_correct': int(correct),
                'test_accuracy': float(test_accuracy),
                'degradation': float(degradation),
                'degradation_pct': float((degradation / train_confidence * 100) if train_confidence > 0 else 0),
                'signal_strength': float(cluster_info.get('signal_strength', 0))
            }

            cluster_results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(cluster_results)

        # Ensure correct data types
        if len(results_df) > 0:
            results_df['cluster_id'] = results_df['cluster_id'].astype(int)
            results_df['expected_label'] = results_df['expected_label'].astype(int)
            results_df['test_samples'] = results_df['test_samples'].astype(int)
            results_df['test_correct'] = results_df['test_correct'].astype(int)
            results_df = results_df.sort_values('signal_strength', ascending=False)

            print(f"\nTested {len(results_df)} actionable clusters")
            print("\nTop 10 Cluster Performance:")
            print("-" * 80)

            for idx, row in results_df.head(10).iterrows():
                status = "PASS" if row['degradation'] < 0.1 else "DEGRADED"
                print(f"Cluster {int(row['cluster_id']):3d} | "
                      f"Expected: {int(row['expected_label'])} | "
                      f"Train: {row['train_confidence']:.2%} | "
                      f"Test: {row['test_accuracy']:.2%} | "
                      f"Samples: {int(row['test_samples']):3d} | "
                      f"Status: {status}")

        return results_df

    def evaluate_by_tier(self, test_df, cluster_results_df):
        """Evaluate performance by confidence tier"""

        print("\n" + "=" * 80)
        print("PERFORMANCE BY TIER")
        print("=" * 80)

        tier_results = {}

        for tier_name, tier_rules in self.rule_catalog['tiers'].items():
            tier_cluster_ids = [int(rule['cluster_id']) for rule in tier_rules]

            # Filter cluster results for this tier
            tier_clusters = cluster_results_df[
                cluster_results_df['cluster_id'].isin(tier_cluster_ids)
            ]

            if len(tier_clusters) > 0:
                tier_results[tier_name] = {
                    'n_clusters': int(len(tier_clusters)),
                    'total_samples': int(tier_clusters['test_samples'].sum()),
                    'avg_train_confidence': float(tier_clusters['train_confidence'].mean()),
                    'avg_test_accuracy': float(tier_clusters['test_accuracy'].mean()),
                    'avg_degradation': float(tier_clusters['degradation'].mean()),
                    'clusters_stable': int((tier_clusters['degradation'] < 0.1).sum())
                }

        print("\nTier Performance Summary:")
        print("-" * 80)

        for tier_name in ['tier_1', 'tier_2', 'tier_3']:
            if tier_name in tier_results:
                result = tier_results[tier_name]
                stability = result['clusters_stable'] / result['n_clusters'] * 100

                print(f"\n{tier_name.upper().replace('_', ' ')}:")
                print(f"  Clusters:         {result['n_clusters']}")
                print(f"  Test Samples:     {result['total_samples']:,}")
                print(f"  Avg Train Conf:   {result['avg_train_confidence']:.2%}")
                print(f"  Avg Test Acc:     {result['avg_test_accuracy']:.2%}")
                print(f"  Avg Degradation:  {result['avg_degradation']:.2%}")
                print(f"  Stable Clusters:  {result['clusters_stable']}/{result['n_clusters']} ({stability:.0f}%)")

        return tier_results

    def identify_degraded_patterns(self, cluster_results_df, threshold=0.15):
        """Identify patterns that degraded significantly"""

        print("\n" + "=" * 80)
        print(f"DEGRADED PATTERNS (>{threshold * 100:.0f}% drop)")
        print("=" * 80)

        degraded = cluster_results_df[cluster_results_df['degradation'] > threshold].copy()
        degraded = degraded.sort_values('degradation', ascending=False)

        # Ensure correct dtypes
        if len(degraded) > 0:
            degraded['cluster_id'] = degraded['cluster_id'].astype(int)
            degraded['test_samples'] = degraded['test_samples'].astype(int)

        if len(degraded) > 0:
            print(f"\nFound {len(degraded)} significantly degraded patterns:")
            print("-" * 80)

            for idx, row in degraded.iterrows():
                print(f"\nCluster {int(row['cluster_id'])}:")
                print(f"  Train Confidence: {row['train_confidence']:.2%}")
                print(f"  Test Accuracy:    {row['test_accuracy']:.2%}")
                print(f"  Degradation:      {row['degradation']:.2%} ({row['degradation_pct']:.1f}%)")
                print(f"  Test Samples:     {int(row['test_samples'])}")
                print("  WARNING: Pattern may have stopped working!")
        else:
            print("\nNo significantly degraded patterns found.")
            print(f"All clusters maintained performance within {threshold * 100:.0f}%")

        return degraded

    def identify_stable_patterns(self, cluster_results_df, threshold=0.05):
        """Identify highly stable patterns"""

        print("\n" + "=" * 80)
        print(f"STABLE PATTERNS (<{threshold * 100:.0f}% drop)")
        print("=" * 80)

        stable = cluster_results_df[
            (cluster_results_df['degradation'] < threshold)
            & (cluster_results_df['train_confidence'] >= 0.9)
        ].copy()
        stable = stable.sort_values('train_confidence', ascending=False)

        # Ensure correct dtypes
        if len(stable) > 0:
            stable['cluster_id'] = stable['cluster_id'].astype(int)
            stable['test_samples'] = stable['test_samples'].astype(int)

        if len(stable) > 0:
            print(f"\nFound {len(stable)} highly stable patterns:")
            print("-" * 80)

            for idx, row in stable.head(15).iterrows():
                print(f"Cluster {int(row['cluster_id']):3d} | "
                      f"Train: {row['train_confidence']:.2%} | "
                      f"Test: {row['test_accuracy']:.2%} | "
                      f"Samples: {int(row['test_samples']):3d} | "
                      f"EXCELLENT")
        else:
            print("\nNo highly stable patterns found with criteria:")
            print("  - Train confidence >= 90%")
            print(f"  - Degradation < {threshold * 100:.0f}%")

        return stable

    def create_performance_visualizations(self, cluster_results_df, output_dir):
        """Create backtest performance visualizations"""

        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        output_dir = Path(output_dir)

        # Create 2x2 visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')

        # 1. Train vs Test Accuracy Scatter
        ax1 = axes[0, 0]

        scatter = ax1.scatter(
            cluster_results_df['train_confidence'],
            cluster_results_df['test_accuracy'],
            s=cluster_results_df['test_samples'] * 2,
            c=cluster_results_df['signal_strength'],
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidths=1
        )

        # Perfect line
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect (no degradation)')

        # Acceptable degradation line (10%)
        ax1.plot([0, 1], [0, 0.9], 'orange', linestyle='--', linewidth=1.5,
                 label='Acceptable (10% degradation)')

        ax1.set_xlabel('Train Confidence', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Test Accuracy', fontweight='bold', fontsize=12)
        ax1.set_title('Train vs Test Performance', fontweight='bold', fontsize=14, pad=15)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.65, 1.05)
        ax1.set_ylim(0.4, 1.05)

        plt.colorbar(scatter, ax=ax1, label='Signal Strength')

        # 2. Degradation Distribution
        ax2 = axes[0, 1]

        degradation_pct = cluster_results_df['degradation'] * 100

        ax2.hist(degradation_pct, bins=20, color='steelblue', alpha=0.7,
                 edgecolor='black', linewidth=1.2)
        ax2.axvline(0, color='green', linestyle='--', linewidth=2, label='No degradation')
        ax2.axvline(10, color='orange', linestyle='--', linewidth=2, label='10% threshold')
        ax2.axvline(15, color='red', linestyle='--', linewidth=2, label='15% critical')

        ax2.set_xlabel('Degradation (%)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Number of Clusters', fontweight='bold', fontsize=12)
        ax2.set_title('Performance Degradation Distribution', fontweight='bold',
                      fontsize=14, pad=15)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Test Accuracy by Signal Strength
        ax3 = axes[1, 0]

        # Sort by signal strength
        sorted_df = cluster_results_df.sort_values('signal_strength', ascending=False)

        colors = ['green' if d < 0.1 else 'orange' if d < 0.15 else 'red'
                  for d in sorted_df['degradation']]

        bars = ax3.bar(range(len(sorted_df)), sorted_df['test_accuracy'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)

        # Add labels on top of bars
        for bar, acc in zip(bars, sorted_df['test_accuracy']):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,  # slightly above bar
                f"{acc:.2f}",  # format accuracy
                ha='center', va='bottom',
                fontsize=7
            )

        ax3.axhline(0.7, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                    label='70% threshold')
        ax3.axhline(0.85, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                    label='85% threshold')
        ax3.axhline(0.95, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                    label='95% threshold')

        ax3.set_xlabel('Cluster (sorted by signal strength)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Test Accuracy', fontweight='bold', fontsize=12)
        ax3.set_title('Test Accuracy by Cluster', fontweight='bold', fontsize=14, pad=15)
        ax3.legend(fontsize=10, loc='lower left')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Sample Size vs Performance
        ax4 = axes[1, 1]

        stable_mask = cluster_results_df['degradation'] < 0.1
        degraded_mask = cluster_results_df['degradation'] >= 0.1

        ax4.scatter(cluster_results_df[stable_mask]['test_samples'],
                    cluster_results_df[stable_mask]['test_accuracy'],
                    c='green', s=100, alpha=0.6, label='Stable (<10% deg)',
                    edgecolors='black', linewidths=1)

        ax4.scatter(cluster_results_df[degraded_mask]['test_samples'],
                    cluster_results_df[degraded_mask]['test_accuracy'],
                    c='red', s=100, alpha=0.6, label='Degraded (>=10% deg)',
                    edgecolors='black', linewidths=1)

        ax4.axhline(0.7, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Test Sample Size', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Test Accuracy', fontweight='bold', fontsize=12)
        ax4.set_title('Sample Size vs Performance', fontweight='bold', fontsize=14, pad=15)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = output_dir / f'backtest_performance_{self.name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\nVisualization saved: {save_path}")

        return fig

    def generate_backtest_report(self, test_df, cluster_results_df, tier_results,
                                 stable_patterns, degraded_patterns, output_dir):
        """Generate comprehensive backtest report"""

        print("\n" + "=" * 80)
        print("GENERATING BACKTEST REPORT")
        print("=" * 80)

        output_dir = Path(output_dir)

        report = {
            'summary': {
                'test_samples': len(test_df),
                'clusters_tested': len(cluster_results_df),
                'stable_patterns': len(stable_patterns),
                'degraded_patterns': len(degraded_patterns),
                'overall_accuracy': accuracy_score(test_df['label'], test_df['predicted_label'])
            },
            'tier_performance': tier_results,
            'cluster_details': cluster_results_df.to_dict('records'),
            'stable_patterns': stable_patterns.to_dict('records') if len(stable_patterns) > 0 else [],
            'degraded_patterns': degraded_patterns.to_dict('records') if len(degraded_patterns) > 0 else []
        }

        def _json_default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            return str(o)

        # Save JSON report
        # with open(output_dir / 'backtest_report.json', 'w', encoding='utf-8') as f:
        #     json.dump(report, f, indent=2, ensure_ascii=False)

        with open(output_dir / f'backtest_report_{self.name}.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)

        # Save CSV of cluster results
        cluster_results_df.to_csv(output_dir / f'cluster_performance_{self.name}.csv',
                                  index=False, encoding='utf-8')

        # Create text summary
        self._create_text_summary(report, output_dir / f'backtest_summary_{self.name}.txt')

        print(f"\nBacktest report saved to: {output_dir}")
        print("  - backtest_report.json")
        print(f"  - cluster_performance_{self.name}.csv")
        print(f"  - backtest_summary_{self.name}.txt")
        print(f"  - backtest_performance_{self.name}.png")

        return report

    def _create_text_summary(self, report, filepath):
        """Create human-readable summary"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BACKTEST SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Overall summary
            f.write("OVERALL RESULTS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Test Samples:      {report['summary']['test_samples']:,}\n")
            f.write(f"Clusters Tested:   {report['summary']['clusters_tested']}\n")
            f.write(f"Overall Accuracy:  {report['summary']['overall_accuracy']:.2%}\n")
            f.write(f"Stable Patterns:   {report['summary']['stable_patterns']}\n")
            f.write(f"Degraded Patterns: {report['summary']['degraded_patterns']}\n\n")

            # Tier summary
            f.write("PERFORMANCE BY TIER:\n")
            f.write("-" * 80 + "\n")
            for tier_name in ['tier_1', 'tier_2', 'tier_3']:
                if tier_name in report['tier_performance']:
                    tier = report['tier_performance'][tier_name]
                    f.write(f"\n{tier_name.upper().replace('_', ' ')}:\n")
                    f.write(f"  Clusters:        {tier['n_clusters']}\n")
                    f.write(f"  Avg Train Conf:  {tier['avg_train_confidence']:.2%}\n")
                    f.write(f"  Avg Test Acc:    {tier['avg_test_accuracy']:.2%}\n")
                    f.write(f"  Stable:          {tier['clusters_stable']}/{tier['n_clusters']}\n")

            # Recommendations
            f.write("\n\nRECOMMENDATIONS:\n")
            f.write("-" * 80 + "\n")

            if report['summary']['stable_patterns'] > 10:
                f.write("* EXCELLENT: You have many stable, reliable patterns\n")
                f.write("* Action: Focus on these stable patterns for live trading\n")
            elif report['summary']['stable_patterns'] > 5:
                f.write("* GOOD: Several patterns maintained performance\n")
                f.write("* Action: Use stable patterns with confidence\n")
            else:
                f.write("* CAUTION: Few patterns maintained train performance\n")
                f.write("* Action: Consider retraining or more feature engineering\n")

            f.write("\n")

            if report['summary']['degraded_patterns'] > 5:
                f.write("* WARNING: Several patterns degraded significantly\n")
                f.write("* Action: Investigate degraded patterns, may be overfit\n")
            else:
                f.write("* GOOD: Most patterns maintained stability\n")

            f.write("\n")


def run_backtest(train_path, test_path, model_path, features_path, rule_catalog_path, output_dir, name):
    """Run backtesting for one dataset (long or short)."""

    name = name.lower().replace(" trades", "")
    print("=" * 80)
    print(f"RUNNING BACKTEST FOR: {output_dir}")
    print("=" * 80)

    # Ensure output directory exists
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

    print(f"    Loaded {len(df):,} total samples")
    print(f"    Loaded {len(features)} features")
    print(f"    Loaded {len(rule_catalog['all_rules'])} rules")

    # === Initialize engine ===
    engine = BacktestEngine(model, features, rule_catalog, df, name)

    # === Execute steps ===
    train_df, test_df = engine.time_based_split(test_size=0.2)
    overall_metrics = engine.evaluate_overall_performance(test_df)
    print(overall_metrics)
    cluster_results = engine.evaluate_by_cluster(test_df)
    tier_results = engine.evaluate_by_tier(test_df, cluster_results)
    stable_patterns = engine.identify_stable_patterns(cluster_results, threshold=0.05)
    degraded_patterns = engine.identify_degraded_patterns(cluster_results, threshold=0.15)
    engine.create_performance_visualizations(cluster_results, output_dir)

    report = engine.generate_backtest_report(
        test_df, cluster_results, tier_results,
        stable_patterns, degraded_patterns, output_dir
    )

    print(f"\n Backtest complete for: {output_dir}")
    print(f"  Accuracy: {report['summary']['overall_accuracy']:.2%}")
    print(f"  Stable Patterns: {report['summary']['stable_patterns']}")
    print(f"  Degraded Patterns: {report['summary']['degraded_patterns']}\n")

    return report


def main():
    """Run backtesting for both LONG and SHORT datasets."""

    test_name = "base_ind_osc"

    configs = [
        {
            "name": "Long Trades",
            "train": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/train_prepared_long.parquet",
            "test": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/test_prepared_long.parquet",
            "features": f"./outputs/test_{test_name}/long_trades/module4_feature_reduction_long_trades/final_features_long.txt",
            "model": f"./outputs/test_{test_name}/long_trades/module5_supervised_clustering_long_trades/decision_tree_model_long.pkl",
            "rules": f"./outputs/test_{test_name}/long_trades/module7_rule_extraction_long_trades/rule_catalog_long.json",
            "output": f"./outputs/test_{test_name}/long_trades/module9_backtesting"
        },
        {
            "name": "Short Trades",
            "train": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/train_prepared_short.parquet",
            "test": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/test_prepared_short.parquet",
            "features": f"./outputs/test_{test_name}/short_trades/module4_feature_reduction_short_trades/final_features_short.txt",
            "model": f"./outputs/test_{test_name}/short_trades/module5_supervised_clustering_short_trades/decision_tree_model_short.pkl",
            "rules": f"./outputs/test_{test_name}/short_trades/module7_rule_extraction_short_trades/rule_catalog_short.json",
            "output": f"./outputs/test_{test_name}/short_trades/module9_backtesting"
        }
    ]

    all_results = {}

    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"STARTING BACKTEST FOR {cfg['name'].upper()}")
        print("=" * 80)

        try:
            result = run_backtest(
                cfg["train"], cfg["test"], cfg["model"],
                cfg["features"], cfg["rules"], cfg["output"], cfg['name']
            )
            all_results[cfg["name"]] = result
        except Exception as e:
            print(f" ERROR during {cfg['name']} backtest: {e}")

    print("\n" + "=" * 80)
    print("ALL BACKTESTS COMPLETED")
    print("=" * 80)

    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result['summary']['overall_accuracy']:.2%}")
        print(f"  Stable: {result['summary']['stable_patterns']}")
        print(f"  Degraded: {result['summary']['degraded_patterns']}")

    return all_results


if __name__ == "__main__":
    results_summary = main()
