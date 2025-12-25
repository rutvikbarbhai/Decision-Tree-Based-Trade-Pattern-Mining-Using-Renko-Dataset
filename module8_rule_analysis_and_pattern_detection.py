#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 4B: Rule Analysis & Pattern Detection

Analyzes extracted rules to find:
1. Common patterns across high-performing clusters
2. Feature importance across rule tiers
3. Condition combinations that predict success
4. Rule conflicts and redundancies
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict

sns.set_style('whitegrid')


class RuleAnalyzer:
    """Analyze trading rules to extract meta-patterns"""

    def __init__(self, rule_catalog, name):
        """
        Args:
            rule_catalog: Dict from module8 with 'all_rules' and 'tiers'
        """
        self.catalog = rule_catalog
        self.name = name
        self.all_rules = rule_catalog['all_rules']

    def analyze_feature_frequency(self):
        """Find which features appear most in rules"""

        print("=" * 80)
        print("FEATURE FREQUENCY ANALYSIS")
        print("=" * 80)

        # Count feature usage by tier
        tier_features = {
            'tier_1': defaultdict(int),
            'tier_2': defaultdict(int),
            'tier_3': defaultdict(int),
            'tier_4': defaultdict(int),
            'all': defaultdict(int)
        }

        for tier_name, tier_rules in self.catalog['tiers'].items():
            for rule in tier_rules:
                for condition in rule['conditions']:
                    feature = condition['feature']
                    tier_features[tier_name][feature] += 1
                    tier_features['all'][feature] += 1

        # Create analysis DataFrames
        analysis = []
        for feature, count in tier_features['all'].items():
            tier_1_count = tier_features['tier_1'][feature]
            tier_2_count = tier_features['tier_2'][feature]
            tier_3_count = tier_features['tier_3'][feature]

            analysis.append({
                'feature': feature,
                'total_count': count,
                'tier_1_count': tier_1_count,
                'tier_2_count': tier_2_count,
                'tier_3_count': tier_3_count,
                'tier_1_pct': (tier_1_count / count * 100) if count > 0 else 0
            })

        df = pd.DataFrame(analysis).sort_values('total_count', ascending=False)

        print("\nTop 15 Most Important Features Across All Rules:")
        print("-" * 80)
        for idx, row in df.head(15).iterrows():
            print("{row['feature'][:50]:50s} | "
                  f"Total: {row['total_count']:2d} | "
                  f"T1: {row['tier_1_count']:2d} | "
                  f"T2: {row['tier_2_count']:2d} | "
                  f"T3: {row['tier_3_count']:2d}")

        return df

    def find_common_patterns(self):
        """Find feature combinations that appear together"""

        print("\n" + "=" * 80)
        print("COMMON PATTERN DETECTION")
        print("=" * 80)

        # Focus on Tier 1 rules (highest confidence)
        tier_1_rules = self.catalog['tiers']['tier_1']

        print(f"\nAnalyzing {len(tier_1_rules)} Tier 1 rules...")

        # Extract feature pairs
        feature_pairs = Counter()

        for rule in tier_1_rules:
            features = [c['feature'] for c in rule['conditions']]

            # Create pairs
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    pair = tuple(sorted([features[i], features[j]]))
                    feature_pairs[pair] += 1

        # Show top patterns
        print("\nTop 10 Feature Pairs in Tier 1 Rules:")
        print("-" * 80)

        for (feat1, feat2), count in feature_pairs.most_common(10):
            pct = (count / len(tier_1_rules)) * 100
            print(f"{count:2d}x ({pct:4.1f}%) | {feat1[:35]:35s} + {feat2[:35]:35s}")

        return feature_pairs

    def analyze_by_signal_direction(self):
        """Compare Win vs loss rule characteristics"""

        print("\n" + "=" * 80)
        print("SIGNAL DIRECTION ANALYSIS")
        print("=" * 80)

        win_rules = [r for r in self.all_rules if r['predicted_class'] == 1]
        loss_rules = [r for r in self.all_rules if r['predicted_class'] == 0]

        print("\n Signal Distribution:")
        print(f"  win signals:  {len(win_rules):2d} rules ({len(win_rules) / len(self.all_rules) * 100:.1f}%)")
        print(f"  loss signals: {len(loss_rules):2d} rules ({len(loss_rules) / len(self.all_rules) * 100:.1f}%)")

        # Average characteristics
        win_avg_conf = np.mean([r['confidence'] for r in win_rules])
        loss_avg_conf = np.mean([r['confidence'] for r in loss_rules])

        win_avg_conditions = np.mean([r['n_conditions'] for r in win_rules])
        loss_avg_conditions = np.mean([r['n_conditions'] for r in loss_rules])

        print("\n Win Signals:")
        print(f"  Average Confidence: {win_avg_conf * 100:.1f}%")
        print(f"  Average Conditions: {win_avg_conditions:.1f}")

        print("\n Loss Signals:")
        print(f"  Average Confidence: {loss_avg_conf * 100:.1f}%")
        print(f"  Average Conditions: {loss_avg_conditions:.1f}")

        # Top features for each direction
        win_features = Counter()
        loss_features = Counter()

        for rule in win_rules:
            for cond in rule['conditions']:
                win_features[cond['feature']] += 1

        for rule in loss_rules:
            for cond in rule['conditions']:
                loss_features[cond['feature']] += 1

        print("\n Top 5 Features for Win Signals:")
        for feat, count in win_features.most_common(5):
            print(f"  {count:2d}x | {feat}")

        print("\n Top 5 Features for Loss Signals:")
        for feat, count in loss_features.most_common(5):
            print(f"  {count:2d}x | {feat}")

        return {
            'win_features': win_features,
            'loss_features': loss_features,
            'win_stats': {
                'count': len(win_rules),
                'avg_confidence': win_avg_conf,
                'avg_conditions': win_avg_conditions
            },
            'loss_stats': {
                'count': len(loss_rules),
                'avg_confidence': loss_avg_conf,
                'avg_conditions': loss_avg_conditions
            }
        }

    def identify_rule_conflicts(self):
        """Find rules with overlapping conditions but different signals"""

        print("\n" + "=" * 80)
        print("RULE CONFLICT DETECTION")
        print("=" * 80)

        conflicts = []

        for i, rule1 in enumerate(self.all_rules):
            for j, rule2 in enumerate(self.all_rules[i + 1:], i + 1):

                # Different signals
                if rule1['predicted_class'] != rule2['predicted_class']:

                    # Check condition overlap
                    features1 = set(c['feature'] for c in rule1['conditions'])
                    features2 = set(c['feature'] for c in rule2['conditions'])

                    overlap = features1.intersection(features2)
                    overlap_pct = len(overlap) / min(len(features1), len(features2))

                    # High overlap = potential conflict
                    if overlap_pct > 0.6:
                        conflicts.append({
                            'cluster_1': rule1['cluster_id'],
                            'cluster_2': rule2['cluster_id'],
                            'signal_1': rule1['predicted_class'],
                            'signal_2': rule2['predicted_class'],
                            'overlap_features': list(overlap),
                            'overlap_pct': overlap_pct,
                            'conf_1': rule1['confidence'],
                            'conf_2': rule2['confidence']
                        })

        if conflicts:
            print(f"\n WARNING:  Found {len(conflicts)} potential rule conflicts")
            print("\nTop 5 Conflicts (features overlap but different signals):")
            print("-" * 80)

            sorted_conflicts = sorted(conflicts, key=lambda x: x['overlap_pct'], reverse=True)

            for conflict in sorted_conflicts[:5]:
                print(f"\nCluster {conflict['cluster_1']} vs Cluster {conflict['cluster_2']}")
                print(f"  Signals: {conflict['signal_1']} vs {conflict['signal_2']}")
                print(f"  Overlap: {conflict['overlap_pct'] * 100:.0f}%")
                print(f"  Confidence: {conflict['conf_1'] * 100:.0f}% vs {conflict['conf_2'] * 100:.0f}%")
                print(f"  Common features: {', '.join(conflict['overlap_features'][:3])}...")
        else:
            print("\n No significant rule conflicts detected")

        return conflicts

    def create_feature_importance_chart(self, feature_freq_df, output_dir):
        """Create visualization of feature importance across tiers"""

        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Top 20 features
        top_features = feature_freq_df.head(20)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Chart 1: Overall frequency
        ax1 = axes[0]
        y_pos = np.arange(len(top_features))

        ax1.barh(y_pos, top_features['total_count'], color='steelblue', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f[:40] for f in top_features['feature']], fontsize=9)
        ax1.set_xlabel('Frequency Across All Rules', fontweight='bold', fontsize=12)
        ax1.set_title('Top 20 Most Important Features', fontweight='bold', fontsize=14, pad=20)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # Chart 2: Tier breakdown
        ax2 = axes[1]

        tier_data = top_features[['tier_1_count', 'tier_2_count', 'tier_3_count']].values

        x = np.arange(len(top_features))
        width = 0.8

        ax2.barh(x, tier_data[:, 0], width, label='Tier 1 (≥95%)',
                 color='#2ecc71', alpha=0.8)
        ax2.barh(x, tier_data[:, 1], width, left=tier_data[:, 0],
                 label='Tier 2 (85-95%)', color='#3498db', alpha=0.8)
        ax2.barh(x, tier_data[:, 2], width,
                 left=tier_data[:, 0] + tier_data[:, 1],
                 label='Tier 3 (70-85%)', color='#95a5a6', alpha=0.8)

        ax2.set_yticks(x)
        ax2.set_yticklabels([f[:40] for f in top_features['feature']], fontsize=9)
        ax2.set_xlabel('Frequency by Tier', fontweight='bold', fontsize=12)
        ax2.set_title('Feature Usage by Confidence Tier', fontweight='bold', fontsize=14, pad=20)
        ax2.legend(loc='lower right', framealpha=0.9, fontsize=10)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        save_path = output_dir / f'feature_importance_analysis_{self.name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\n Saved visualization: {save_path}")

        return fig

    def create_analysis_report(self, output_dir):
        """Create comprehensive analysis report"""

        output_dir = Path(output_dir)

        print("\n" + "=" * 80)
        print("CREATING ANALYSIS REPORT")
        print("=" * 80)

        # Run all analyses
        feature_freq = self.analyze_feature_frequency()
        patterns = self.find_common_patterns()
        print(patterns)
        direction_analysis = self.analyze_by_signal_direction()
        conflicts = self.identify_rule_conflicts()

        # Create visualization
        self.create_feature_importance_chart(feature_freq, output_dir)

        # Save detailed analysis
        analysis_dict = {
            'feature_frequency': feature_freq.to_dict('records'),
            'direction_analysis': {
                'win_stats': direction_analysis['win_stats'],
                'loss_stats': direction_analysis['loss_stats'],
                'win_top_features': dict(direction_analysis['win_features'].most_common(10)),
                'loss_top_features': dict(direction_analysis['loss_features'].most_common(10))
            },
            'conflicts': conflicts
        }

        with open(output_dir / f'rule_analysis_{self.name}.json', 'w') as f:
            json.dump(analysis_dict, f, indent=2)

        print(f"\n Saved analysis: {output_dir / f'rule_analysis_{self.name}.json'}")

        # Key insights report
        self._create_insights_report(output_dir / f'key_insights_{self.name}.txt',
                                     feature_freq, direction_analysis)

        print(f" Saved insights: {output_dir / f'key_insights_{self.name}.txt'}")

        return analysis_dict

    def _create_insights_report(self, filepath, feature_freq, direction_analysis):
        """Create key insights summary"""

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KEY INSIGHTS FROM RULE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # 1. Most critical features
            f.write(" MOST CRITICAL FEATURES (appear in most rules):\n")
            f.write("-" * 80 + "\n")
            for idx, row in feature_freq.head(5).iterrows():
                f.write(f"{idx + 1}. {row['feature']}\n")
                f.write(f"   Used in {row['total_count']} rules "
                        f"({row['tier_1_count']} Tier 1, "
                        f"{row['tier_2_count']} Tier 2, "
                        f"{row['tier_3_count']} Tier 3)\n\n")

            # 2. Signal bias
            f.write("\n SIGNAL DIRECTION INSIGHTS:\n")
            f.write("-" * 80 + "\n")

            win_count = direction_analysis['win_stats']['count']
            loss_count = direction_analysis['loss_stats']['count']
            total = win_count + loss_count

            f.write(f"Win bias: {win_count}/{total} rules ({win_count / total * 100:.1f}%)\n")
            f.write(f"Loss bias: {loss_count}/{total} rules ({loss_count / total * 100:.1f}%)\n\n")

            if win_count > loss_count * 1.5:
                f.write("  WARNING: Strong Win bias detected\n")
                f.write("   Consider: Are we missing Loss opportunities?\n\n")
            elif loss_count > win_count * 1.5:
                f.write("  WARNING: Strong Loss bias detected\n")
                f.write("   Consider: Are we too conservative on Wins?\n\n")
            else:
                f.write(" BALANCED: Good distribution of Win/Loss signals\n\n")

            # 3. Actionable recommendations
            f.write("\n ACTIONABLE RECOMMENDATIONS:\n")
            f.write("-" * 80 + "\n")
            f.write("1. Focus on top 5 features for feature engineering improvements\n")
            f.write("2. Monitor Tier 1 rules closely - these are your edge\n")
            f.write("3. Validate signal bias matches your market assumptions\n")
            f.write("4. Use Tier 3 rules with additional confirmation signals\n")


def run_rule_analysis(rule_catalog_path, output_dir, name):
    """Run full rule analysis & pattern detection for one dataset (long or short)."""

    name = name.lower().replace(" trades", "")
    print("=" * 80)
    print(f"RUNNING RULE ANALYSIS FOR: {output_dir}")
    print("=" * 80)

    if not Path(rule_catalog_path).exists():
        print(f" Missing rule catalog file: {rule_catalog_path}")
        return None, None

    with open(rule_catalog_path, 'r') as f:
        catalog = json.load(f)

    print(f" Loaded {catalog['summary']['total_rules']} rules")

    analyzer = RuleAnalyzer(catalog, name)
    analysis = analyzer.create_analysis_report(output_dir)

    print(f"\n Rule analysis complete for: {output_dir}")
    print("-" * 80)
    return analyzer, analysis


def main():
    """Run rule analysis and pattern detection for both LONG and SHORT datasets."""
    test_name = "base_ind_osc"

    configs = [
        {
            "name": "Long Trades",
            "catalog": f"./outputs/test_{test_name}/long_trades/module7_rule_extraction_long_trades/rule_catalog_long.json",
            "output": f"./outputs/test_{test_name}/long_trades/module8_rule_analysis_and_pattern_detection_long_trades"
        },
        {
            "name": "Short Trades",
            "catalog": f"./outputs/test_{test_name}/short_trades/module7_rule_extraction_short_trades/rule_catalog_short.json",
            "output": f"./outputs/test_{test_name}/short_trades/module8_rule_analysis_and_pattern_detection_short_trades"
        }
    ]

    all_results = {}

    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"RUNNING {cfg['name'].upper()} RULE ANALYSIS")
        print("=" * 80)

        analyzer, analysis = run_rule_analysis(cfg["catalog"], cfg["output"], cfg['name'])
        if analyzer is not None:
            all_results[cfg["name"]] = {"analyzer": analyzer, "analysis": analysis}

    print("\n" + "=" * 80)
    print("ALL RULE ANALYSES COMPLETE ")
    print("=" * 80)

    for name, result in all_results.items():
        print(f"\n{name}: {len(result['analysis']['feature_frequency'])} features analyzed")

    return all_results


if __name__ == "__main__":
    results_summary = main()
