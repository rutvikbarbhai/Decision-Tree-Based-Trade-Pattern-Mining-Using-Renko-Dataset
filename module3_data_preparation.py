#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 3: Dataset Preparation
Properly splits data TEMPORALLY and prepares features WITHOUT data leakage
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from data.storage import WasabiStorageSystem


class DatasetPreparator:

    """Prepare clean train/test datasets with proper feature engineering"""
    exclude_cols = {
        'trade_idx', 'name', 'symbol', 'signal', 'strategy',
        'entry_date', 'exit_date', 'entry_cost', 'exit_cost',
        'pnl', 'label', 'max_loss', 'max_profit', 'margin',
        'rrr', 'return', 'minutes', 'spot_return', 'credit',
        'no_bricks', 'brick_size', 'jump_size', 'reversal_bricks',
        'bricks_duration', 'no_bricks_no_delay', 'jump_size_no_delay',
        'reversal_bricks_no_delay', 'bricks_duration_no_delay',
        'desc', 'delay_loss', 'quadrant', 'open_cost', 'close_cost',
        'days', 'weekly_return', 'daily_return', 'weekly_spot_return',
        'daily_spot_return', 'hit_ratio', 'open_date', 'close_date',
        'pos_pnl', 'neg_pnl', 'total_pnl', 'drawdown',
        'theoretical_pnl', 'negative_contribution',
        'pnl_without_negative_contribution', 'category',
        'datetime', 'adjusted_close', 'Unnamed: 0', "wasabi_key", "t", "bar_timestamp"
    }

    def __init__(self, data_path, output_dir='./outputs/module3_prepared_data'):
        self.output_dir = Path(output_dir)
        self.mode = "long" if "long" in str(self.output_dir).lower() else "short"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.train_df = None
        self.test_df = None
        self.feature_engineering_artifacts = {}

    def load_data_from_wasabi(self, tickers=None, trade_type="long", test_name=None):
        print("=" * 80)
        print(f"STEP 1: LOADING WASABI DATA ({trade_type.upper()} | test={test_name})")
        print("=" * 80)

        storage = WasabiStorageSystem(
            path='ml_nonquad/tsfresh_outputs',
            bucket='nufintech-ai-common'
        )

        # pattern matching ALL tickers
        pattern = f"*_{trade_type}_final_trades_df_test_{test_name}.pkl"
        print(f"Searching for files matching: {pattern}")

        matched_files = storage.ls("", pattern)
        # matched_files = matched_files[0:2]

        print(f"Found {len(matched_files)} files")

        if not matched_files:
            raise ValueError("No matching Wasabi files found.")

        all_dfs = []

        master_cols = None  # Stores column order of first DF

        for key in matched_files:
            try:
                df = storage.read_pickle(key)
                print(f"Loaded: {key}")

                current_cols = list(df.columns)

                if master_cols is None:
                    master_cols = current_cols
                    print(f"\nMaster column template set from: {key}")
                    print(f"Column count: {len(master_cols)}")
                    print("Column order preserved.\n")

                else:
                    missing = [c for c in master_cols if c not in current_cols]
                    extra = [c for c in current_cols if c not in master_cols]

                    if missing or extra:
                        print("\n COLUMN MISMATCH DETECTED!")
                        print(f"File: {key}")

                        if missing:
                            print(f"  Missing columns: {missing}")
                        if extra:
                            print(f"  Extra columns: {extra}")

                        raise SystemExit(
                            f"Exiting because file '{key}' has missing or extra columns."
                        )

                    if current_cols != master_cols:
                        df = df.reindex(columns=master_cols)

                # Add key after all checks
                df["wasabi_key"] = key

                all_dfs.append(df)

            except Exception as e:
                raise SystemExit(f"Failed to load {key}: {e}")

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.copy()
        combined_df.insert(10, 'label', (combined_df['pnl'] > 0).astype(int))
        combined_df = combined_df.rename(columns={"symbol_x": "symbol"})
        combined_df = combined_df.drop(columns=['symbol_y'], errors="ignore")

        print(f"Combined rows: {len(combined_df)} | Columns: {len(combined_df.columns)}")

        return combined_df

    def train_test_split_data(self, df, test_size=0.2, random_state=42, temporal=True):
        """
        Split data FIRST before any feature engineering

        uses temporal split by default for time-series data

        Args:
            df: Full dataset
            test_size: Proportion for test set
            random_state: Random seed (only used if temporal=False)
            temporal: If True, split by time (RECOMMENDED for stocks)
                      If False, use stratified random split
        """
        print("\n" + "=" * 80)
        print("STEP 2: TRAIN/TEST SPLIT")
        print("=" * 80)

        if 'label' not in df.columns:
            raise ValueError("'label' column not found in dataset")

        print("\nClass distribution in full data:")
        print(df['label'].value_counts())
        print(df['label'].value_counts(normalize=True))

        if temporal:
            print("\nUsing TEMPORAL SPLIT")

            if 'entry_date' not in df.columns:
                raise ValueError("'entry_date' column required for temporal split")

            df_sorted = df.sort_values('entry_date').reset_index(drop=True)

            split_idx = int(len(df_sorted) * (1 - test_size))

            train_df = df_sorted.iloc[:split_idx].copy()
            test_df = df_sorted.iloc[split_idx:].copy()

            print(f"\nTRAIN PERIOD: {train_df['entry_date'].min()} to {train_df['entry_date'].max()}")
            print(f" TEST PERIOD:  {test_df['entry_date'].min()} to {test_df['entry_date'].max()}")

            train_max = train_df['entry_date'].max()
            test_min = test_df['entry_date'].min()

            # Validate no temporal overlap
            if train_max >= test_min:
                raise ValueError(
                    f"TEMPORAL SPLIT VALIDATION FAILED!\n"
                    f"Train max date ({train_max}) >= Test min date ({test_min})\n"
                    f"This indicates temporal overlap between train/test sets."
                )

            time_gap = (test_min - train_max).days
            print(f" Temporal validation passed: {time_gap} day gap between train and test")

            if time_gap > 30:
                print(f" Note: {time_gap} day gap between train and test")

        else:
            print("\n Using STRATIFIED RANDOM SPLIT")
            print("    WARNING: Not recommended for time-series data!")

            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df['label']
            )

        print(f"\n Train set: {len(train_df):,} samples ({len(train_df) / len(df) * 100:.1f}%)")
        print(f" Test set: {len(test_df):,} samples ({len(test_df) / len(df) * 100:.1f}%)")

        print("\nTrain class distribution:")
        print(train_df['label'].value_counts())
        print(train_df['label'].value_counts(normalize=True))

        print("\nTest class distribution:")
        print(test_df['label'].value_counts())
        print(test_df['label'].value_counts(normalize=True))

        self.train_df = train_df.reset_index(drop=True)
        self.train_df = self.train_df.drop(columns=["t", "trade_idx"])
        self.test_df = test_df.reset_index(drop=True)
        self.test_df = self.test_df.drop(columns=["t", "trade_idx"])

        return self.train_df, self.test_df

    def apply_feature_drops(self, features_to_drop_path):
        """Apply feature drops from Module 3 analysis"""
        print("\n" + "=" * 80)
        print("STEP 3: APPLYING FEATURE DROPS")
        print("=" * 80)

        features_to_drop_path = Path(features_to_drop_path)
        if not features_to_drop_path.exists():
            print("\n  features_to_drop.txt not found. Skipping feature drops.")
            print("   This is OK on first run - we'll skip feature drops for now.")
            return self.train_df, self.test_df

        with open(features_to_drop_path, 'r') as f:
            features_to_drop = [line.strip() for line in f.readlines()]

        print(f"\nLoaded {len(features_to_drop)} features to drop")

        existing_drops = [f for f in features_to_drop if f in self.train_df.columns]
        missing_drops = [f for f in features_to_drop if f not in self.train_df.columns]

        if missing_drops:
            print(f"\n  {len(missing_drops)} features not found in dataset")

        if existing_drops:
            print(f"\n Dropping {len(existing_drops)} features from train set")
            self.train_df = self.train_df.drop(columns=existing_drops)

            print(f" Dropping {len(existing_drops)} features from test set")
            self.test_df = self.test_df.drop(columns=existing_drops)

            print(f"\nRemaining features: {len(self.train_df.columns)}")
        else:
            print("\n No features to drop")

        return self.train_df, self.test_df

    def save_datasets(self):

        """Save prepared datasets and artifacts"""
        print("\n" + "=" * 80)
        print("STEP 5: SAVING PREPARED DATASETS")
        print("=" * 80)

        print("\nCleaning boolean-like object columns in TRAIN/TEST...")

        def clean_boolean_like_columns(df):
            for col in df.columns:
                if df[col].dtype == "object":
                    uniq = set(df[col].dropna().unique())
                    # if object col contains only boolean-ish values
                    if uniq <= {True, False} or uniq <= {True, False, None}:
                        print(f"  -> Converting {col} from object to bool")
                        df[col] = df[col].fillna(False).astype(bool)
            return df

        self.train_df = clean_boolean_like_columns(self.train_df)
        self.test_df = clean_boolean_like_columns(self.test_df)

        train_path = self.output_dir / f'train_prepared_{self.mode}.parquet'
        test_path = self.output_dir / f'test_prepared_{self.mode}.parquet'

        train_path_csv = self.output_dir / f'train_prepared_{self.mode}.csv'
        test_path_csv = self.output_dir / f'test_prepared_{self.mode}.csv'

        self.train_df.to_parquet(train_path, index=False)
        self.test_df.to_parquet(test_path, index=False)

        self.train_df.to_csv(train_path_csv, index=False)
        self.test_df.to_csv(test_path_csv, index=False)

        print(f"\n Saved train data: {train_path}")
        print(f"  - Shape: {self.train_df.shape}")

        print(f"\n Saved test data: {test_path}")
        print(f"  - Shape: {self.test_df.shape}")

        # Build clean feature list AFTER boolean-fix
        feature_cols = [c for c in self.train_df.columns if c not in self.exclude_cols]

        features_path = self.output_dir / f'prepared_features_{self.mode}.txt'
        with open(features_path, 'w') as f:
            f.write('\n'.join(sorted(feature_cols)))

        print(f" Saved feature list: {features_path}")

        return train_path, test_path


def prepare_dataset(output_dir, trade_type, test_name, tickers=None):
    """
    Centralized wrapper for preparing datasets from Wasabi.
    """
    preparator = DatasetPreparator(
        data_path=None,
        output_dir=output_dir
    )

    # LOAD FROM WASABI
    df = preparator.load_data_from_wasabi(
        tickers=tickers,
        trade_type=trade_type,
        test_name=test_name
    )

    # TRAIN/TEST SPLIT
    train_df, test_df = preparator.train_test_split_data(df)

    # SAVE
    train_path, test_path = preparator.save_datasets()

    return train_path, test_path


def main():
    test_name = "base_ind_osc"

    tasks = [
        {
            "trade_type": "long",
            "test_name": test_name,
            "output_dir": f"./outputs/test_{test_name}/long_trades/module3_prepared_data_long_trades"
        },
        {
            "trade_type": "short",
            "test_name": test_name,
            "output_dir": f"./outputs/test_{test_name}/short_trades/module3_prepared_data_short_trades"
        }
    ]

    for t in tasks:
        print(f"\n=== RUNNING TASK ({t['trade_type'].upper()} | test={t['test_name']}) ===")
        train_path, test_path = prepare_dataset(
            output_dir=t["output_dir"],
            trade_type=t["trade_type"],
            test_name=t["test_name"],
            tickers=None
        )

        print(f"\n DONE: {t['output_dir']}")
        print(f"   Train: {train_path}")
        print(f"   Test : {test_path}")


if __name__ == "__main__":
    main()
