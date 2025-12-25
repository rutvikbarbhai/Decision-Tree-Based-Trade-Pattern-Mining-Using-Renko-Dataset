#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module 4: Feature Reduction
works with pre-split data from Module 0
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path
import json


class QuickWinsFeatureReducer:
    """Applies quick wins to reduce features efficiently"""

    def __init__(self, train_df, features, mode, target_col='label', exclude_cols=None):
        self.train_df = train_df
        self.features = features.copy()
        self.target_col = target_col
        self.exclude_cols = exclude_cols or [
            'trade_idx', 'name', 'symbol', 'signal', 'strategy', 'entry_date', 'exit_date', 'entry_cost',
            'exit_cost', 'pnl', 'label', 'max_loss', 'max_profit', 'margin', 'rrr', 'return', 'minutes',
            'spot_return', 'credit', 'no_bricks', 'brick_size', 'jump_size', 'reversal_bricks',
            'bricks_duration', 'no_bricks_no_delay', 'jump_size_no_delay', 'reversal_bricks_no_delay',
            'bricks_duration_no_delay', 'desc', 'delay_loss', 'quadrant', 'open_cost', 'close_cost', 'days',
            'weekly_return', 'daily_return', 'weekly_spot_return', 'daily_spot_return', 'hit_ratio',
            'open_date', 'close_date', 'pos_pnl', 'neg_pnl', 'total_pnl', 'drawdown', 'theoretical_pnl',
            'negative_contribution', 'pnl_without_negative_contribution', 'category', 'datetime',
            'adjusted_close', 'Unnamed: 0',
        ]
        self.history = []

    def log_step(self, step_name, features_before, features_after, dropped):
        """Log each reduction step"""
        self.history.append({
            'step': step_name,
            'features_before': features_before,
            'features_after': features_after,
            'dropped': dropped,
            'reduction': len(dropped)
        })

    def consolidate_time_windows(self, prefer='5d'):
        """Keep only one time window per metric"""
        print("\n" + "=" * 70)
        print("STEP 1: TIME WINDOW CONSOLIDATION")
        print("=" * 70)

        features_before = len(self.features)
        feature_groups = {}

        for feature in self.features:
            # base = feature.replace('_5d', '').replace('_10d', '').replace('_20d', '')
            base = feature.replace('_w5', '').replace('_w10', '')
            if base != feature:
                if base not in feature_groups:
                    feature_groups[base] = []
                feature_groups[base].append(feature)

        keep = []
        drop = []
        processed_bases = set()

        for base, variants in feature_groups.items():
            processed_bases.add(base)
            if len(variants) <= 1:
                keep.extend(variants)
            else:
                preferred = [f for f in variants if f'_{prefer}' in f]
                if preferred:
                    keep.append(preferred[0])
                    drop.extend([f for f in variants if f not in preferred])
                else:
                    keep.append(variants[0])
                    drop.extend(variants[1:])

        # Only add features without time windows if not already processed
        for feature in self.features:
            base = feature.replace('_5d', '').replace('_10d', '').replace('_20d', '')
            if base == feature and base not in processed_bases:
                keep.append(feature)

        self.features = list(set(keep))

        print(f"Before: {features_before} features")
        print(f"After: {len(self.features)} features")
        print(f"Dropped: {len(drop)} redundant time windows")

        self.log_step('time_window_consolidation', features_before, len(self.features), drop)
        return drop

    def domain_based_filtering(self, verify_importance=True):
        """Remove low-value features based on domain knowledge"""
        print("\n" + "=" * 70)
        print("STEP 2: DOMAIN-BASED FILTERING")
        print("=" * 70)

        features_before = len(self.features)

        drop_candidates = {
            'coefficient_variation': [f for f in self.features if 'coefficient_variation' in f],
            'volatility_redundant': [f for f in self.features if 'volatility' in f and 'pre_' in f],
            'distance_measures': [f for f in self.features if 'distance_' in f],
        }

        all_candidates = []
        for features in drop_candidates.values():
            all_candidates.extend(features)
        all_candidates = list(set(all_candidates))

        print(f"Found {len(all_candidates)} drop candidates")

        protected = []
        if verify_importance and all_candidates:
            print("\nVerifying importance on TRAINING data only...")
            X = self.train_df[self.features].fillna(0)
            y = self.train_df[self.target_col]

            rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            # Protect top 30 features
            top_features = importance_df.head(30)['feature'].tolist()
            protected = [f for f in all_candidates if f in top_features]

            if protected:
                print(f"  Protecting {len(protected)} important candidates from dropping")

        # Drop non-protected candidates
        final_drops = [f for f in all_candidates if f not in protected]
        self.features = [f for f in self.features if f not in final_drops]

        print(f"\nBefore: {features_before} features")
        print(f"After: {len(self.features)} features")
        print(f"Dropped: {len(final_drops)} low-value features")
        if protected:
            print(f"Protected: {len(protected)} important features")

        self.log_step('domain_filtering', features_before, len(self.features), final_drops)
        return final_drops

    def variance_filter(self, threshold=1e-5):
        print("\n" + "=" * 70)
        print("STEP: VARIANCE THRESHOLD FILTER")
        print("=" * 70)

        print("\nCleaning boolean-like object columns...")

        for col in self.train_df.columns:
            if self.train_df[col].dtype == "object":
                uniq = set(self.train_df[col].dropna().unique())
                # If unique values are subset of {True, False, None}
                if uniq <= {True, False} or uniq <= {True, False, None}:
                    print(f"  -> Converting {col} from object to bool")
                    self.train_df[col] = self.train_df[col].fillna(False).astype(bool)

        total_features_before = len(self.features)

        numeric_cols = self.train_df.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        # Keep only numeric-cols that are listed as features
        numeric_cols = [c for c in numeric_cols if c in self.features]

        print(f"\nNumeric cols Before: {len(numeric_cols)} features")

        X = self.train_df[numeric_cols].fillna(0)

        vt = VarianceThreshold(threshold=threshold)
        vt.fit(X)

        keep = X.columns[vt.get_support()].tolist()

        drop = [c for c in self.features if c not in keep]

        print(f"After variance filter: {len(keep)} features (dropped {len(drop)})")

        # Debug: show exactly which were non-numeric BEFORE replacement
        non_numeric = set(self.features) - set(numeric_cols)
        if non_numeric:
            print("\nNon-numeric features (excluded before variance filter):")
            for col in sorted(non_numeric):
                print("   -", col)
        else:
            print("\nNon-numeric feature columns: set()")

        self.features = keep

        self.log_step(
            step_name="variance_filter",
            features_before=total_features_before,
            features_after=len(self.features),
            dropped=drop,
        )

        return drop

    def correlation_pruning(self, threshold=0.97):
        print("\n" + "=" * 70)
        print("STEP: CORRELATION PRUNING")
        print("=" * 70)

        features_before = len(self.features)

        X = self.train_df[self.features].fillna(0)

        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        drop = [column for column in upper.columns if any(upper[column] > threshold)]
        keep = [c for c in self.features if c not in drop]

        self.features = keep

        print(f"Before: {features_before} features")
        print(f"After correlation pruning: {len(self.features)} features (dropped {len(drop)})")

        self.log_step("correlation_pruning", features_before, len(self.features), drop)
        return drop

    def rank_by_importance(self, top_n=50):
        """Final importance-based ranking to get to target size"""
        print("\n" + "=" * 70)
        print(f"STEP 3: IMPORTANCE-BASED FINAL SELECTION (Top {top_n})")
        print("=" * 70)

        features_before = len(self.features)

        if features_before <= top_n:
            print(f"Already at or below target ({features_before} <= {top_n})")
            return []

        print("Training Random Forest for final importance ranking...")
        X = self.train_df[self.features].fillna(0)
        y = self.train_df[self.target_col]

        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        top_features = importance_df.head(top_n)['feature'].tolist()
        dropped = importance_df.tail(features_before - top_n)['feature'].tolist()

        print("\nTop 10 most important features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:50s}: {row['importance']:.4f}")

        self.features = top_features

        print(f"\nBefore: {features_before} features")
        print(f"After: {len(self.features)} features")
        print(f"Dropped: {len(dropped)} least important features")

        self.log_step('importance_ranking', features_before, len(self.features), dropped)
        return dropped

    def get_summary(self):
        """Print final summary"""
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        initial = self.history[0]['features_before']
        final = len(self.features)

        print(f"\nInitial features: {initial}")
        print(f"Final features: {final}")
        print(f"Total reduction: {initial - final} ({(initial - final) / initial * 100:.1f}%)")

        print("\nReduction by step:")
        for step in self.history:
            print(f"  {step['step']:30s}: -{step['reduction']:3d} features")

        return {
            'initial_features': initial,
            'final_features': final,
            'final_feature_list': self.features,
            'history': self.history
        }


def run_feature_reduction(train_data_path, initial_features_path, output_dir, target_features=50):
    """Reusable function for feature reduction"""

    print("=" * 80)
    print(f"FEATURE REDUCTION: {output_dir}")
    print("=" * 80)

    if not Path(train_data_path).exists():
        print(f"ERROR: {train_data_path} not found!")
        return None, None

    print("\nLoading TRAINING data...")
    train_df = pd.read_parquet(train_data_path)
    # train_df = train_df.drop(columns=["t", "trade_idx"])
    print(f" Loaded {len(train_df):,} training samples")

    with open(initial_features_path, 'r') as f:
        initial_features = [line.strip() for line in f.readlines()]
    print(f" Starting with {len(initial_features)} features")

    mode = "long" if "long" in str(train_data_path).lower() else "short"

    reducer = QuickWinsFeatureReducer(train_df, initial_features, mode, target_col='label')

    reducer.variance_filter(threshold=1e-5)
    reducer.correlation_pruning(threshold=0.97)
    reducer.rank_by_importance(top_n=target_features)

    summary = reducer.get_summary()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / f'final_features_{mode}.txt', 'w') as f:
        f.write('\n'.join(sorted(summary['final_feature_list'])))

    with open(output_dir / f'reduction_summary_{mode}.json', 'w') as f:
        save_summary = summary.copy()
        save_summary.pop('history')
        json.dump(save_summary, f, indent=2)

    print(f"\n Results saved to: {output_dir}")
    print(f"  - final_features.txt ({len(summary['final_feature_list'])} features)")
    print("  - reduction_summary.json")

    return reducer, summary


def main():
    """Run feature reduction for both long and short trades"""

    test_name = "base_ind_osc"

    configs = [
        {
            "train_data_path": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/train_prepared_long.parquet",
            "initial_features_path": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades/prepared_features_long.txt",
            "output_dir": f"./outputs/test_{test_name}/long_trades/module4_feature_reduction_long_trades",
        },
        {
            "train_data_path": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/train_prepared_short.parquet",
            "initial_features_path": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades/prepared_features_short.txt",
            "output_dir": f"./outputs/test_{test_name}/short_trades/module4_feature_reduction_short_trades",
        },
    ]

    all_results = {}
    for cfg in configs:
        reducer, summary = run_feature_reduction(
            cfg["train_data_path"],
            cfg["initial_features_path"],
            cfg["output_dir"],
            target_features=100
        )
        all_results[cfg["output_dir"]] = summary

    print("\n" + "=" * 80)
    print(" ALL FEATURE REDUCTIONS COMPLETE")
    print("=" * 80)

    for output_dir, summary in all_results.items():
        print(f"\n→ {output_dir}")
        print(f"  Final features: {len(summary['final_feature_list'])}")

    return all_results


if __name__ == "__main__":
    all_results = main()
