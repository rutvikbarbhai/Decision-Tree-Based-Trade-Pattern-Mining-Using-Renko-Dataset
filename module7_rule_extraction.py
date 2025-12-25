#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 4.1: Extract Trading Rules from Decision Tree Clusters
"""

import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.tree import _tree


class TradingRuleExtractor:
    """Extract and analyze decision tree rules for trading"""

    def __init__(self, model, feature_names, actionable_clusters, name):
        self.model = model
        self.feature_names = feature_names
        self.actionable_clusters = actionable_clusters
        self.name = name
        self.tree = model.tree_
        self.rules = []

        # Get valid leaf node IDs
        children_left = self.tree.children_left
        children_right = self.tree.children_right

        self.valid_leaf_ids = set([
            i for i in range(self.tree.node_count)
            if children_left[i] == _tree.TREE_LEAF and children_right[i] == _tree.TREE_LEAF
        ])

        # Normalize cluster key names (handle 'cluster' vs 'cluster_id')
        self._normalize_cluster_keys()

        # Validate inputs
        self._validate_inputs()

    def _normalize_cluster_keys(self):
        """Normalize cluster key names from Module 3 output"""
        if len(self.actionable_clusters) == 0:
            return

        # Check if we need to rename keys
        first_cluster = self.actionable_clusters[0]

        if 'cluster' in first_cluster and 'cluster_id' not in first_cluster:
            # Rename 'cluster' to 'cluster_id' for consistency
            for c in self.actionable_clusters:
                c['cluster_id'] = c.pop('cluster')
            print("[Normalization] Renamed 'cluster' key to 'cluster_id'")

        key_mappings = {
            'dominant_class': 'predicted_class',
            'purity': 'confidence'
        }

        for c in self.actionable_clusters:
            for old_key, new_key in key_mappings.items():
                if old_key in c and new_key not in c:
                    c[new_key] = c[old_key]

    def _validate_inputs(self):
        """Validate that inputs are consistent"""
        n_leaves = len(self.valid_leaf_ids)

        print(f"\n{'=' * 70}")
        print("VALIDATION")
        print(f"{'=' * 70}")
        print("Tree Structure:")
        print(f"  Total nodes: {self.tree.node_count}")
        print(f"  Leaf nodes: {n_leaves}")
        print(f"  Valid leaf IDs: {min(self.valid_leaf_ids)} to {max(self.valid_leaf_ids)}")
        print("\nActionable Clusters:")
        print(f"  Count: {len(self.actionable_clusters)}")

        if len(self.actionable_clusters) == 0:
            print("\n WARNING: No actionable clusters provided!")
            print("   This might mean:")
            print("   1. Module 3 didn't find any high-confidence clusters")
            print("   2. Wrong file loaded (check path)")
            print("   3. Need to lower purity/size thresholds in Module 3")
            return

        if len(self.actionable_clusters) > 0:
            first_cluster = self.actionable_clusters[0]
            print(f"\nFirst cluster keys: {list(first_cluster.keys())}")

        # Check for invalid cluster IDs
        invalid_clusters = [
            c for c in self.actionable_clusters
            if c['cluster_id'] not in self.valid_leaf_ids
        ]

        if invalid_clusters:
            print(f"\n ERROR: Found {len(invalid_clusters)} invalid cluster IDs:")

            raise ValueError(
                f"Found {len(invalid_clusters)} invalid cluster IDs. "

            )

        print(f"\n Validation passed: All {len(self.actionable_clusters)} clusters are valid")

    def extract_all_rules(self):
        """Extract rules for all actionable clusters"""
        print("\n" + "=" * 70)
        print("EXTRACTING DECISION RULES")
        print("=" * 70)

        if len(self.actionable_clusters) == 0:
            print("\n  No actionable clusters to extract rules from")
            return []

        # Calculate signal_strength for all clusters
        for cluster_info in self.actionable_clusters:
            cluster_info['signal_strength'] = self._calculate_signal_strength(cluster_info)

        # Sort by signal_strength
        sorted_clusters = sorted(
            self.actionable_clusters,
            key=lambda x: x['signal_strength'],
            reverse=True
        )

        print(f"\nExtracting rules for {len(sorted_clusters)} clusters...")

        for i, cluster_info in enumerate(sorted_clusters, 1):
            cluster_id = cluster_info['cluster_id']
            try:
                rule = self._extract_rule_for_cluster(cluster_id, cluster_info)
                self.rules.append(rule)

                if i % 10 == 0:
                    print(f"  Processed {i}/{len(sorted_clusters)} clusters...")

            except Exception as e:
                print(f"  Warning: Could not extract rule for cluster {cluster_id}: {e}")
                continue

        print(f"\n Extracted {len(self.rules)} trading rules")
        return self.rules

    def _extract_rule_for_cluster(self, leaf_id, cluster_info):
        """Extract the decision path for a specific leaf node"""
        path = self._find_path_to_leaf(leaf_id)

        conditions = []
        for node_id, direction in path:
            if self.tree.feature[node_id] != _tree.TREE_UNDEFINED:
                feature_idx = self.tree.feature[node_id]

                # Validate feature index
                if feature_idx >= len(self.feature_names):
                    print(f"  ⚠️  Warning: Invalid feature index {feature_idx}")
                    continue

                feature_name = self.feature_names[feature_idx]
                threshold = self.tree.threshold[node_id]
                operator = '<=' if direction == 'left' else '>'

                conditions.append({
                    'feature': feature_name,
                    'operator': operator,
                    'threshold': float(threshold),
                    'readable': f"{feature_name} {operator} {threshold:.4f}"
                })

        # Get the correct keys
        predicted_class = cluster_info.get('predicted_class', cluster_info.get('dominant_class'))
        confidence = cluster_info.get('confidence', cluster_info.get('purity'))

        rule = {
            'cluster_id': int(leaf_id),
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'size': int(cluster_info['size']),
            'conditions': conditions,
            'n_conditions': len(conditions),
            'signal_strength': float(cluster_info['signal_strength'])
        }

        return rule

    def _find_path_to_leaf(self, leaf_id):
        """Find the path from root to a specific leaf node"""
        n_nodes = self.tree.node_count
        children_left = self.tree.children_left
        children_right = self.tree.children_right

        parent = [-1] * n_nodes
        direction = [None] * n_nodes

        # Build parent-child relationships
        for node_id in range(n_nodes):
            left = children_left[node_id]
            right = children_right[node_id]

            if left != _tree.TREE_LEAF:
                parent[left] = node_id
                direction[left] = 'left'

            if right != _tree.TREE_LEAF:
                parent[right] = node_id
                direction[right] = 'right'

        # Traverse from leaf to root
        path = []
        current = leaf_id

        while parent[current] != -1:
            path.append((parent[current], direction[current]))
            current = parent[current]

        path.reverse()
        return path

    def _calculate_signal_strength(self, cluster_info):
        """Calculate a signal strength score (0-10)"""

        confidence = cluster_info.get('confidence', cluster_info.get('purity', 0.5))
        size = cluster_info.get('size', 0)

        # 70% weight on confidence, 30% on size
        confidence_score = confidence * 7
        size_score = min(size / 100, 1.0) * 3

        return round(confidence_score + size_score, 2)

    def create_rule_catalog(self):
        """Create a comprehensive rule catalog"""
        print("\n" + "=" * 70)
        print("CREATING RULE CATALOG")
        print("=" * 70)

        tier_1 = [r for r in self.rules if r['confidence'] >= 0.95]
        tier_2 = [r for r in self.rules if 0.85 <= r['confidence'] < 0.95]
        tier_3 = [r for r in self.rules if 0.70 <= r['confidence'] < 0.85]
        # temporarily added
        tier_4 = [r for r in self.rules if 0.60 <= r['confidence'] < 0.70]

        catalog = {
            'summary': {
                'total_rules': len(self.rules),
                'tier_1_rules': len(tier_1),
                'tier_2_rules': len(tier_2),
                'tier_3_rules': len(tier_3),
                'tier_4_rules': len(tier_4)
            },
            'tiers': {
                'tier_1': tier_1,
                'tier_2': tier_2,
                'tier_3': tier_3,
                'tier_4': tier_4
            },
            'all_rules': self.rules
        }

        print("\nRule Distribution:")
        print(f"  Tier 1 (≥95% confidence): {len(tier_1)} rules")
        print(f"  Tier 2 (85-95% confidence): {len(tier_2)} rules")
        print(f"  Tier 3 (70-85% confidence): {len(tier_3)} rules")
        print(f"  Tier 4 (60-70% confidence): {len(tier_4)} rules")

        return catalog

    def print_human_readable_rules(self, top_n=10):
        """Print rules in human-readable format"""
        print("\n" + "=" * 70)
        print(f"TOP {min(top_n, len(self.rules))} TRADING RULES")
        print("=" * 70)

        if len(self.rules) == 0:
            print("\n  No rules to display")
            return

        top_rules = sorted(
            self.rules,
            key=lambda x: x['signal_strength'],
            reverse=True
        )[:top_n]

        for i, rule in enumerate(top_rules, 1):
            self._print_rule(i, rule)

    def _print_rule(self, rank, rule):
        """Print a single rule in readable format"""
        print(f"\n{'=' * 70}")
        print(f"RANK #{rank} | CLUSTER {rule['cluster_id']}")
        print(f"{'=' * 70}")

        signal = "BUY" if rule['predicted_class'] == 1 else "AVOID"
        confidence_pct = rule['confidence'] * 100

        print(f"\nSIGNAL: {signal}")
        print(f"Confidence: {confidence_pct:.1f}%")
        print(f"Signal Strength: {rule['signal_strength']}/10")
        print(f"Sample Size: {rule['size']:,} trades")

        print(f"\nCONDITIONS ({rule['n_conditions']} total):")
        print("-" * 70)

        for j, cond in enumerate(rule['conditions'], 1):
            feature = cond['feature']
            operator = cond['operator']
            threshold = cond['threshold']

            # Clean feature name
            feature_clean = feature.replace('_', ' ').title()
            if len(feature_clean) > 50:
                feature_clean = feature_clean[:47] + "..."

            print(f"  {j}. {feature_clean}")
            print(f"     -> {operator} {threshold:.4f}")

        print("\nINTERPRETATION:")
        self._interpret_rule(rule)

    def _interpret_rule(self, rule):
        """Provide trading interpretation of the rule"""
        conditions = rule['conditions']
        signal = "bullish" if rule['predicted_class'] == 1 else "bearish"

        # Analyze condition types
        momentum_conditions = [c for c in conditions if 'momentum' in c['feature'].lower()]
        volatility_conditions = [c for c in conditions if
                                 'volatility' in c['feature'].lower() or 'vol' in c['feature'].lower()]
        position_conditions = [c for c in conditions if
                               'position' in c['feature'].lower() or 'range' in c['feature'].lower()]

        print(f"  • This is a {signal.upper()} pattern")

        if momentum_conditions:
            print(f"  • Momentum-driven ({len(momentum_conditions)} momentum conditions)")

        if volatility_conditions:
            low_vol = [c for c in volatility_conditions if c['operator'] == '<=']
            high_vol = [c for c in volatility_conditions if c['operator'] == '>']

            if len(low_vol) > len(high_vol):
                vol_dir = "low"
            elif len(high_vol) > len(low_vol):
                vol_dir = "high"
            else:
                vol_dir = "mixed"

            print(f"  • Requires {vol_dir} volatility environment")

        if position_conditions:
            print(f"  • Position-dependent ({len(position_conditions)} position conditions)")

        # Confidence assessment
        if rule['confidence'] >= 0.95:
            print("  • HIGH CONVICTION - Very reliable pattern")
        elif rule['confidence'] >= 0.85:
            print("  • GOOD CONVICTION - Reliable pattern")
        else:
            print("  • MODERATE CONVICTION - Use with confirmation")

    def export_rules(self, output_dir):
        """Export rules to multiple formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("\n" + "=" * 70)
        print("EXPORTING RULES")
        print("=" * 70)

        # 1. JSON export
        catalog = self.create_rule_catalog()

        with open(output_dir / f'rule_catalog_{self.name}.json', 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"\nExported: rule_catalog_{self.name}.json")

        # 2. CSV export
        if len(self.rules) > 0:
            rules_df = self._rules_to_dataframe()
            rules_df.to_csv(output_dir / f'rules_summary_{self.name}.csv', index=False, encoding='utf-8')
            print(f" Exported: rules_summary_{self.name}.csv")

        # 3. Detailed text report
        self._create_text_report(output_dir / f'rules_detailed_{self.name}.txt')
        print(f" Exported: rules_detailed_{self.name}.txt")

        # 4. Quick reference guide
        self._create_quick_reference(output_dir / f'rules_quick_reference_{self.name}.txt')
        print(f" Exported: rules_quick_reference{self.name}.txt")

        print(f"\n All files saved to: {output_dir}")

    def _rules_to_dataframe(self):
        """Convert rules to DataFrame for analysis"""
        rows = []
        for rule in self.rules:
            row = {
                'cluster_id': rule['cluster_id'],
                'predicted_class': rule['predicted_class'],
                'signal': 'BUY' if rule['predicted_class'] == 1 else 'AVOID',
                'confidence': rule['confidence'],
                'confidence_pct': f"{rule['confidence'] * 100:.1f}%",
                'size': rule['size'],
                'signal_strength': rule['signal_strength'],
                'n_conditions': rule['n_conditions'],
                'conditions_summary': ' AND '.join([c['readable'][:50] for c in rule['conditions'][:3]])
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _create_text_report(self, filepath):
        """Create detailed text report"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TRADING RULES - DETAILED REPORT\n")
            f.write("=" * 80 + "\n\n")

            if len(self.rules) == 0:
                f.write("No rules extracted.\n")
                return

            sorted_rules = sorted(
                self.rules,
                key=lambda x: x['signal_strength'],
                reverse=True
            )

            for i, rule in enumerate(sorted_rules, 1):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"RULE #{i} | CLUSTER {rule['cluster_id']}\n")
                f.write(f"{'=' * 80}\n\n")

                signal = "BUY" if rule['predicted_class'] == 1 else "AVOID"
                f.write(f"Signal: {signal}\n")
                f.write(f"Confidence: {rule['confidence'] * 100:.1f}%\n")
                f.write(f"Signal Strength: {rule['signal_strength']}/10\n")
                f.write(f"Sample Size: {rule['size']} trades\n")

                f.write(f"\nConditions ({rule['n_conditions']} total):\n")
                for j, cond in enumerate(rule['conditions'], 1):
                    f.write(f"  {j}. {cond['readable']}\n")

                f.write("\n")

    def _create_quick_reference(self, filepath):
        """Create quick reference guide for trading desk"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("QUICK REFERENCE GUIDE - TOP TRADING SIGNALS\n")
            f.write("=" * 80 + "\n\n")

            if len(self.rules) == 0:
                f.write("No rules available.\n")
                return

            tier_1 = [r for r in self.rules if r['confidence'] >= 0.95]
            tier_1_sorted = sorted(tier_1, key=lambda x: x['signal_strength'], reverse=True)

            f.write(f"TIER 1 SIGNALS (≥95% Confidence) - {len(tier_1)} rules\n")
            f.write("-" * 80 + "\n\n")

            if len(tier_1) == 0:
                f.write("No Tier 1 signals found.\n\n")
            else:
                for rule in tier_1_sorted:
                    signal = "BUY" if rule['predicted_class'] == 1 else "AVOID"
                    f.write(f"Cluster {rule['cluster_id']:3d} | {signal:5s} | "
                            f"{rule['confidence'] * 100:.0f}% | "
                            f"Size: {rule['size']:4d} | "
                            f"Strength: {rule['signal_strength']:.1f}/10\n")

                    for cond in rule['conditions'][:3]:
                        f.write(f"  * {cond['readable']}\n")
                    f.write("\n")


def run_rule_extraction(model_path, features_path, actionable_path, output_dir, name):
    """Run rule extraction pipeline for one dataset (long or short)."""

    name = name.lower().replace(" trades", "")
    print("=" * 80)
    print(f"RUNNING RULE EXTRACTION FOR: {output_dir}")
    print("=" * 80)

    # Validate files
    for path in [model_path, features_path, actionable_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    # Load model, features, and actionable clusters
    model = joblib.load(model_path)
    with open(features_path, 'r', encoding='utf-8') as f:
        features = [line.strip() for line in f.readlines()]
    with open(actionable_path, 'r', encoding='utf-8') as f:
        actionable_clusters = json.load(f)

    print(f"  Model leaves: {model.get_n_leaves()}")
    print(f"  Features: {len(features)}")
    print(f"  Actionable clusters: {len(actionable_clusters)}")

    # Create extractor
    extractor = TradingRuleExtractor(model, features, actionable_clusters, name)

    # Extract rules
    rules = extractor.extract_all_rules()
    if len(rules) == 0:
        print("\n WARNING: No rules extracted — check actionable cluster thresholds.\n")
        return extractor, rules

    # Print top few
    extractor.print_human_readable_rules(top_n=5)

    # Export rules
    extractor.export_rules(output_dir)

    # Summary
    catalog = extractor.create_rule_catalog()
    print(f"   Tier 1: {catalog['summary']['tier_1_rules']} rules")
    print(f"   Tier 2: {catalog['summary']['tier_2_rules']} rules")
    print(f"   Tier 3: {catalog['summary']['tier_3_rules']} rules")
    print(f"\n Done: {len(rules)} rules extracted and exported to {output_dir}")
    return extractor, rules


def main():
    """Run trading rule extraction for both LONG and SHORT datasets."""

    test_name = "base_ind_osc"
    configs = [
        {
            "name": "Long Trades",
            "model": f"./outputs/test_{test_name}/long_trades/module5_supervised_clustering_long_trades/decision_tree_model_long.pkl",
            "features": f"./outputs/test_{test_name}/long_trades/module4_feature_reduction_long_trades/final_features_long.txt",
            "actionable": f"./outputs/test_{test_name}/long_trades/module6_cluster_analysis_long_trades/actionable_clusters_train_long.json",
            "output": f"./outputs/test_{test_name}/long_trades/module7_rule_extraction_long_trades"
        },
        {
            "name": "Short Trades",
            "model": f"./outputs/test_{test_name}/short_trades/module5_supervised_clustering_short_trades/decision_tree_model_short.pkl",
            "features": f"./outputs/test_{test_name}/short_trades/module4_feature_reduction_short_trades/final_features_short.txt",
            "actionable": f"./outputs/test_{test_name}/short_trades/module6_cluster_analysis_short_trades/actionable_clusters_train_short.json",
            "output": f"./outputs/test_{test_name}/short_trades/module7_rule_extraction_short_trades"
        }
    ]

    results_summary = {}

    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"RUNNING {cfg['name'].upper()} RULE EXTRACTION")
        print("=" * 80)

        extractor, rules = run_rule_extraction(
            cfg["model"], cfg["features"], cfg["actionable"], cfg["output"], cfg['name']
        )

        results_summary[cfg["name"]] = {
            "extractor": extractor,
            "n_rules": len(rules)
        }

    print("\n" + "=" * 80)
    print("ALL RULE EXTRACTIONS COMPLETE")
    print("=" * 80)

    for name, result in results_summary.items():
        print(f"\n{name}: {result['n_rules']} rules extracted")

    return results_summary


if __name__ == "__main__":
    results_summary = main()
