"""
Generating combined_file for long and short trades

Usage:
    python module1_comb_file_creation.py --input_folder /path/to/csv/folder
    --trade_pattern "*_2023-04-04_2025-09-30_5min_long.renko_trades.all.v3.4l.top_1.csv" for long trades
    or
    --trade_pattern "*_2023-04-04_2025-09-30_5min_short.renko_trades.all.v3.4l.top_1.csv" for short trades
"""
import warnings
from datetime import datetime
# import pickle
# import io
import re
import argparse
from data.storage import WasabiStorageSystem
import glob
import pandas as pd
import os
from core.settings import Settings
settings = Settings()

warnings.filterwarnings('ignore')


class CombinedFileGenerator:
    """Combined file generation"""

    def __init__(self, input_folder, trade_pattern):
        self.input_folder = input_folder
        self.trade_pattern = trade_pattern
        self.combined_df = None

    def load_and_combine_symbols(self):
        """Load all symbol CSV files"""

        print("\n" + "=" * 70)
        print("STEP 1: LOADING DATA")
        print("=" * 70)

        pattern = os.path.join(self.input_folder, self.trade_pattern)
        csv_files = glob.glob(pattern)
        # csv_files = [f for f in csv_files if "BMY" in f or "PFE" in f]
        # csv_files = csv_files[0:10]

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found matching pattern in {self.input_folder}")

        print(f"\nFound {len(csv_files)} symbol files")

        all_trades = []
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            symbol = filename.split('_')[0]

            df = pd.read_csv(csv_file)
            df['symbol'] = symbol

            for col in ['entry_date', 'exit_date', 'open_date', 'close_date', 'datetime']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            all_trades.append(df)

            if len(all_trades) <= 10 or len(all_trades) % 10 == 0:
                print(f"  Loaded {len(all_trades)} symbols...")

        self.combined_df = pd.concat(all_trades, ignore_index=True)
        self.combined_df = self.combined_df.sort_values(
            ['symbol', 'entry_date']).reset_index(drop=True)

        print(f"\n{'─'*70}")
        print("COMBINED DATASET:")
        print(f"  Total trades: {len(self.combined_df):,}")
        print(f"  Symbols: {self.combined_df['symbol'].nunique()}")
        print(f"{'─'*70}\n")

        return self

    def upload_to_wasabi(self, trade_pattern: str):
        """Parse filename, save csv & pickle, upload both to Wasabi"""

        match = re.search(r"_(\d{4}-\d{2}-\d{2}_.+?_5min_(short|long))", trade_pattern)
        if not match:
            raise ValueError(
                f"Cannot parse trade pattern for filename: {trade_pattern}")

        core = match.group(1)

        parts = core.split('_')
        date_range = "_".join(parts[0:3])
        side = parts[3]

        if side == "short":
            outname = f"comb_short_renko_trades_{date_range}"
        else:
            outname = f"comb_long_renko_trades_{date_range}"

        csv_filename = outname + '.csv'
        pkl_filename = outname + '.pkl'

        wasabi_prefix = "ml_nonquad/inputs"

        ws = WasabiStorageSystem(path=wasabi_prefix)

        ws.write_csv(self.combined_df, symbol="", file=csv_filename, index=False)
        print(f"[Uploaded CSV] s3://{ws.bucket_name}/{wasabi_prefix}/{csv_filename}")

        ws.write_pickle(self.combined_df, symbol="", file=pkl_filename)
        print(f"[Uploaded PKL] s3://{ws.bucket_name}/{wasabi_prefix}/{pkl_filename}")

    def run(self):
        """Execute fixed tsfresh extraction pipeline"""

        print("\n" + "=" * 70)
        print(" FIXED TSFRESH EXTRACTION (No Duplicates)")
        print("=" * 70)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            self.load_and_combine_symbols()
            self.upload_to_wasabi(self.trade_pattern)

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
        description='Generating combined file for both long and short trades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""combined file generation"""
    )

    parser.add_argument(
        '--input_folder', '-i',
        type=str,
        required=True,
        help='Folder containing symbol CSV files'
    )

    parser.add_argument(
        '--trade_pattern', '-t',
        type=str,
        required=True,
        help='Pattern of trades that needs to be picked (long or short)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        return 1

    extractor = CombinedFileGenerator(args.input_folder, args.trade_pattern)
    success = extractor.run()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
