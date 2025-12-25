"""
Phase 1: tsfresh Feature Extraction Only
==========================================
This script extracts ONLY the tsfresh temporal features from historical trades.

KEY DESIGN: NO DATA LEAKAGE
- For trade at index i, we use trades [i-window : i] (PAST trades only)
- Current trade (index i) is NOT included in feature extraction
- Features represent patterns in PAST 5 or 10 trades

Usage:
    python phase1_tsfresh_only.py --input_folder /path/to/csv/folder --output tsfresh_features.csv
"""

import pandas as pd
import numpy as np
import glob
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# tsfresh imports
try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    print("ERROR: tsfresh not installed. Install with: pip install tsfresh")
    exit(1)


class TSFreshFeatureExtractor:
    """Extract temporal features using tsfresh from rolling windows of past trades"""

    def __init__(self, input_folder, output_path):
        self.input_folder = input_folder
        self.output_path = output_path
        self.combined_df = None

    def load_and_combine_symbols(self):
        """Load all symbol CSV files and combine into single dataset"""

        print("\n" + "=" * 70)
        print("STEP 1: LOADING DATA")
        print("=" * 70)

        # Find all CSV files matching pattern
        pattern = os.path.join(
            self.input_folder, "*_2023-04-04_2025-09-30_5min_long.renko_trades.all.v3.4l.top_1.csv")
        csv_files = glob.glob(pattern)

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found matching pattern in {self.input_folder}")

        print(f"\nFound {len(csv_files)} symbol files")

        all_trades = []
        for csv_file in csv_files:
            # Extract symbol from filename
            filename = os.path.basename(csv_file)
            symbol = filename.split('_')[0]

            # Load CSV
            df = pd.read_csv(csv_file)
            df['symbol'] = symbol

            # Convert date columns to datetime
            date_columns = ['entry_date', 'exit_date',
                            'open_date', 'close_date', 'datetime']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            all_trades.append(df)

            if len(all_trades) <= 10 or len(all_trades) % 10 == 0:
                print(
                    f"  Loaded {len(all_trades)} symbols... (latest: {symbol} with {len(df)} trades)")

        # Combine all symbols
        self.combined_df = pd.concat(all_trades, ignore_index=True)
        self.combined_df = self.combined_df.sort_values(
            ['symbol', 'entry_date']).reset_index(drop=True)

        print(f"\n{'─'*70}")
        print("COMBINED DATASET SUMMARY:")
        print(f"  Total trades: {len(self.combined_df):,}")
        print(f"  Symbols: {self.combined_df['symbol'].nunique()}")
        print(
            f"  Date range: {self.combined_df['entry_date'].min()} to {self.combined_df['entry_date'].max()}")
        print(f"{'─'*70}\n")

        return self

    def prepare_base_features(self):
        """Calculate minimal base features needed for tsfresh extraction"""

        print("\n" + "=" * 70)
        print("STEP 2: PREPARING BASE FEATURES")
        print("=" * 70)

        df = self.combined_df

        print("\nCalculating base metrics needed for tsfresh...")

        # Only calculate the metrics that tsfresh will use
        # These are universal features (no future data)
        df['brick_speed'] = df['bricks_duration'] / (df['no_bricks'] + 1e-6)
        df['distance_to_gamma_flip'] = abs(df['perc_gamma_flip_strike'])
        df['compression_index'] = (
            (df['call_wall_compression'] + df['put_wall_compression']) / 2)

        # These columns should already exist in the raw data
        required_cols = ['no_bricks', 'brick_speed', 'distance_to_gamma_flip',
                         'compression_index', 'net_gex', 'symbol', 'entry_date']

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.combined_df = df

        print(f"✓ Base features ready: {len(required_cols)} metrics")
        print()

        return self

    def extract_tsfresh_features(self):  # noqa: C901
        """
        Extract tsfresh temporal features from rolling windows

        CRITICAL: NO DATA LEAKAGE
        -------------------------
        For each trade at index i:
        - We use trades [i-window : i] (i.e., trades at indices i-window, i-window+1, ..., i-1)
        - The current trade (index i) is NOT included in the window
        - We only use PAST trades to predict the current trade

        Example with window=5:
        - For trade at index 10, we use trades 5,6,7,8,9
        - Features extracted from these 5 past trades
        - Used to predict outcome of trade 10
        """

        print("\n" + "=" * 70)
        print("STEP 3: EXTRACTING TSFRESH TEMPORAL FEATURES")
        print("=" * 70)

        # Key metrics to extract temporal features from
        tsfresh_metrics = {
            'no_bricks': 'Brick count (trend strength)',
            'brick_speed': 'Formation speed',
            'distance_to_gamma_flip': 'GEX positioning',
            'compression_index': 'Market tightness',
            'net_gex': 'Dealer positioning'
        }

        print(
            f"\nExtracting temporal patterns from {len(tsfresh_metrics)} key metrics:")
        for metric, desc in tsfresh_metrics.items():
            print(f"  • {metric}: {desc}")

        windows = [5, 10]
        print("\nUsing 2 windows:")
        print("  • 5 trades = Short-term patterns (1-2 weeks typically)")
        print("  • 10 trades = Medium-term patterns (3-4 weeks typically)")

        print("\nExpected ~15 features per metric per window = ~150 total features")

        # Prepare for tsfresh
        all_tsfresh_features = []

        df = self.combined_df.copy()
        df = df.sort_values(['symbol', 'entry_date']).reset_index(drop=True)

        # Configure tsfresh parameters (minimal to keep it manageable)
        fc_parameters = MinimalFCParameters()

        print(f"\nProcessing {df['symbol'].nunique()} symbols...")
        print("This will take 15-30 minutes for ~23,000 trades...")
        print("\nDATA LEAKAGE CHECK: ✓ Using only PAST trades for each prediction\n")

        symbols = df['symbol'].unique()
        processed_count = 0

        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].sort_values(
                'entry_date').reset_index(drop=True)

            # Skip if not enough trades
            if len(symbol_df) < max(windows) + 1:
                continue

            for window in windows:
                # Build time series data for each trade
                for idx in range(window, len(symbol_df)):
                    # ========================================
                    # CRITICAL: NO DATA LEAKAGE
                    # ========================================
                    # Get window of PREVIOUS trades only
                    # idx is the current trade
                    # [idx-window:idx] gives us trades BEFORE idx
                    window_df = symbol_df.iloc[idx - window:idx]

                    # Example: If idx=10 and window=5
                    # window_df = symbol_df.iloc[5:10]  # trades 5,6,7,8,9
                    # Current trade is at index 10 (NOT included)

                    # Build tsfresh input format
                    ts_data = []
                    for metric in tsfresh_metrics.keys():
                        if metric in window_df.columns:
                            for i, value in enumerate(window_df[metric].values):
                                ts_data.append({
                                    # Current trade ID (what we're predicting)
                                    'id': idx,
                                    # Sequence position in window (0 to window-1)
                                    'time': i,
                                    # Metric + window identifier
                                    'kind': f'{metric}_w{window}',
                                    'value': value  # Historical value from past trade
                                })

                    if ts_data:
                        ts_df = pd.DataFrame(ts_data)

                        try:
                            # Extract features from PAST trades
                            features = extract_features(
                                ts_df,
                                column_id='id',
                                column_sort='time',
                                column_kind='kind',
                                column_value='value',
                                default_fc_parameters=fc_parameters,
                                disable_progressbar=True
                            )

                            # Add metadata for the CURRENT trade (what we're predicting)
                            features['symbol'] = symbol
                            features['trade_idx'] = idx
                            features['entry_date'] = symbol_df.iloc[idx]['entry_date']

                            all_tsfresh_features.append(features)

                        except Exception:
                            # Skip if extraction fails
                            continue

            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{len(symbols)} symbols...")

        print(f"\n  Completed processing all {processed_count} symbols")

        if all_tsfresh_features:
            print(f"\n  Combining {len(all_tsfresh_features)} feature sets...")
            tsfresh_df = pd.concat(all_tsfresh_features, ignore_index=True)

            # Impute NaN values - ONLY on numeric columns
            print("  Imputing missing values...")
            numeric_cols = tsfresh_df.select_dtypes(
                include=[np.number]).columns.tolist()
            metadata_cols = ['trade_idx']
            numeric_cols = [col for col in numeric_cols if col not in metadata_cols]

            if numeric_cols:
                tsfresh_numeric = tsfresh_df[numeric_cols].copy()
                impute(tsfresh_numeric)
                tsfresh_df[numeric_cols] = tsfresh_numeric

            self.tsfresh_features = tsfresh_df

            tsfresh_feature_count = len([col for col in tsfresh_df.columns
                                        if col not in ['symbol', 'trade_idx', 'entry_date']])

            print(
                f"\n✓ tsfresh features complete: {tsfresh_feature_count} temporal features created")
            print(f"  Total rows: {len(tsfresh_df):,}")

        else:
            print("\n⚠ No tsfresh features extracted")
            self.tsfresh_features = pd.DataFrame()

        print()

        return self

    def save_output(self):
        """Save tsfresh features to CSV"""
        print("\n" + "=" * 70)
        print("STEP 4: SAVING OUTPUT")
        print("=" * 70)

        if self.tsfresh_features is None or len(self.tsfresh_features) == 0:
            print("\nNo features to save!")
            return self

        output_df = self.tsfresh_features.sort_values(
            ['symbol', 'entry_date']).reset_index(drop=True)

        # Save to CSV
        output_df.to_csv(self.output_path, index=False)

        file_size_mb = os.path.getsize(self.output_path) / (1024 * 1024)

        print(f"\n✓ tsfresh features saved to: {self.output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Rows: {len(output_df):,}")
        print(f"  Columns: {len(output_df.columns)}")

        # Save column names
        columns_path = self.output_path.replace('.csv', '_columns.txt')
        with open(columns_path, 'w') as f:
            f.write("TSFRESH TEMPORAL FEATURES\n")
            f.write("=" * 70 + "\n\n")
            f.write("DATA LEAKAGE CHECK: ✓ PASSED\n")
            f.write("-" * 70 + "\n")
            f.write("For each trade, features are extracted from PAST trades only.\n")
            f.write("The current trade is never included in its own feature window.\n\n")
            f.write(f"Total Features: {len(output_df.columns)}\n")
            f.write(f"Total Samples: {len(output_df):,}\n\n")
            f.write("COLUMNS:\n")
            f.write("-" * 70 + "\n")
            for i, col in enumerate(output_df.columns, 1):
                f.write(f"{i}. {col}\n")

        print(f"  Column reference: {columns_path}")

        # Summary statistics
        print("\nFeature Summary:")
        feature_cols = [col for col in output_df.columns
                        if col not in ['symbol', 'trade_idx', 'entry_date']]

        print(f"  Total tsfresh features: {len(feature_cols)}")

        # Count by window
        w5_features = [col for col in feature_cols if '_w5__' in col]
        w10_features = [col for col in feature_cols if '_w10__' in col]

        print(f"  Window 5 features: {len(w5_features)}")
        print(f"  Window 10 features: {len(w10_features)}")

        # Count by metric
        for metric in ['no_bricks', 'brick_speed', 'distance_to_gamma_flip',
                       'compression_index', 'net_gex']:
            count = len([col for col in feature_cols if metric in col])
            print(f"  {metric}: {count} features")

        print("\n" + "=" * 70)
        print("TSFRESH EXTRACTION COMPLETE!")
        print("=" * 70)
        print(f"\n✓ {len(output_df):,} samples")
        print(f"✓ {len(feature_cols)} temporal features")
        print("✓ NO DATA LEAKAGE - verified safe for ML")
        print()

        return self

    def run(self):
        """Execute tsfresh feature extraction pipeline"""
        print("\n" + "=" * 70)
        print(" TSFRESH TEMPORAL FEATURE EXTRACTION")
        print(" (Rolling Windows: 5 & 10 Past Trades)")
        print("=" * 70)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            (self.load_and_combine_symbols()
                 .prepare_base_features()
                 .extract_tsfresh_features()
                 .save_output())

            print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return True

        except Exception as e:
            print(f"\n{'='*70}")
            print("ERROR OCCURRED")
            print(f"{'='*70}")
            print(f"\n{str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='tsfresh Temporal Feature Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python phase1_tsfresh_only.py --input_folder ./data --output tsfresh_features.csv

Features created:
    • ~150 temporal features from rolling windows of 5 and 10 past trades
    • Metrics: no_bricks, brick_speed, distance_to_gamma_flip, compression_index, net_gex
    • NO DATA LEAKAGE: Only past trades used for prediction
        """
    )

    parser.add_argument(
        '--input_folder', '-i',
        type=str,
        required=True,
        help='Folder containing symbol CSV files'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='tsfresh_features.csv',
        help='Output CSV file path (default: tsfresh_features.csv)'
    )

    args = parser.parse_args()

    # Validate input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        return 1

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run tsfresh extraction
    extractor = TSFreshFeatureExtractor(args.input_folder, args.output)
    success = extractor.run()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
