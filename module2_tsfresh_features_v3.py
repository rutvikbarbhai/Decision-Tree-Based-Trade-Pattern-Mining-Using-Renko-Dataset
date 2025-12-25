"""
Phase 1 tsfresh Feature Extraction
===========================================================
Original problem:
- Window loop created separate rows for w5 and w10 features
- Trades ≥10 appeared twice (once for each window)
- Result: 2x rows, duplicate symbol-dates

 approach:
- Process each trade once
- Combine w5 and w10 features into single row
- No duplicates!

Usage:
   python module2_tsfresh_features_v3.py --type long --ticker BMY --config config/pipeline_config.txt

"""
from core.settings import Settings
from data.storage import WasabiStorageSystem
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

settings = Settings()

CONFIG_PIPELINE = '''
[BASE_FEATURES]



[TSFRESH_METRICS_JSON]
{
  "net_gex": "net_gex",
  "z20_net_gex": "z20_net_gex",
  "neutral_gex": "neutral_gex",
  "gamma_flip_strike": "gamma_flip_strike",
  "gamma_flip_strike_perc": "gamma_flip_strike_perc",
  "gamma_flip_strike_perc_exp_mean": "gamma_flip_strike_perc_exp_mean",
  "gamma_flip_strike_low_vs_hist": "gamma_flip_strike_low_vs_hist",
  "gamma_flip_strike_high_vs_hist": "gamma_flip_strike_high_vs_hist",
  "gamma_flip_strike_not_near": "gamma_flip_strike_not_near",
  "max_call_wall_strike": "max_call_wall_strike",
  "max_call_wall_strike_perc": "max_call_wall_strike_perc",
  "max_call_wall_strike_perc_exp_mean": "max_call_wall_strike_perc_exp_mean",
  "max_call_wall_strike_low_vs_hist": "max_call_wall_strike_low_vs_hist",
  "max_call_wall_strike_high_vs_hist": "max_call_wall_strike_high_vs_hist",
  "max_call_wall_strike_not_near": "max_call_wall_strike_not_near",
  "min_put_wall_strike": "min_put_wall_strike",
  "min_put_wall_strike_perc": "min_put_wall_strike_perc",
  "min_put_wall_strike_perc_exp_mean": "min_put_wall_strike_perc_exp_mean",
  "min_put_wall_strike_low_vs_hist": "min_put_wall_strike_low_vs_hist",
  "min_put_wall_strike_high_vs_hist": "min_put_wall_strike_high_vs_hist",
  "min_put_wall_strike_not_near": "min_put_wall_strike_not_near",
  "neutral_zone": "neutral_zone",
  "call_wall_compression": "call_wall_compression",
  "put_wall_compression": "put_wall_compression",
  "short_gamma_expansion": "short_gamma_expansion",
  "gamma_valley": "gamma_valley",
  "wall_to_wall_compression": "wall_to_wall_compression"
}
'''


# tsfresh imports
try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    print("ERROR: tsfresh not installed. Install with: pip install tsfresh")
    exit(1)


def apply_dynamic_features(df: pd.DataFrame, formulas: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Applies 'name = expr' formulas with DataFrame.eval(engine='python').
    Supports numpy via @np.
    """
    if not formulas:
        print("ℹ No dynamic base features provided.")
        return df

    print(f"Applying {len(formulas)} base feature formulas...")
    for name, expr in formulas:
        try:
            df.eval(f"`{name}` = {expr}", engine="python",
                    local_dict={"np": np}, inplace=True)
            print(f"   {name}")
        except Exception as e:
            print(f"   {name} = {expr}  -> {e}")
    return df


def _split_config_sections(lines):
    mode = None
    base, json_lines = [], []
    for line in lines.strip().splitlines():
        if not line or line.startswith("#"):
            continue

        if line.upper() == "[BASE_FEATURES]":
            mode = "base"
            continue
        if line.upper() == "[TSFRESH_METRICS_JSON]":
            mode = "json"
            continue

        if mode == "base":
            base.append(line)
        elif mode == "json":
            json_lines.append(line)

    return base, json_lines


def _parse_base_features(lines):
    formulas = []
    for line in lines:
        if "=" in line:
            name, expr = line.split("=", 1)
            formulas.append((name.strip(), expr.strip()))
    return formulas


def _parse_metrics_json(lines):
    try:
        joined = "\n".join(lines)
        obj = json.loads(joined)
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        print(f"WARNING: Failed to parse TSFRESH_METRICS_JSON: {e}")
        return {}


def load_pipeline_config(config: str):
    base_lines, json_lines = _split_config_sections(config)

    formulas = _parse_base_features(base_lines)
    metrics = _parse_metrics_json(json_lines)

    return formulas, metrics


class TSFreshExtractor:
    """ tsfresh extraction without duplicates"""

    def __init__(self, ticker, type, path, suffix, test):
        self.ticker = ticker
        self.type = type
        self.path = path
        self.suffix = suffix
        self.test = test
        self.windows = [5, 10]
        self.base_formulas, self.tsfresh_metrics = load_pipeline_config(CONFIG_PIPELINE)

        self.combined_df = None

        print(
            f"Loaded {len(self.base_formulas)} base feature formulas from pipeline")
        print(
            f"Loaded {len(self.tsfresh_metrics)} tsfresh metrics from pipeline")

    def load_from_wasabi(self):
        print("\n" + "=" * 70)
        print(f"STEP 1: EXTRACTING {self.type.upper()} DATA FROM WASABI")
        print("=" * 70)
        storage = WasabiStorageSystem(path='renko/trades', bucket='nufintech-data-analysis')
        keys = storage.ls(
            '',
            f'{self.ticker}/{self.path}/'
            f'{self.ticker.upper()}*_*{self.type}{self.suffix}'
        )
        print(f"Found keys: {len(keys)}")
        dfs = [storage.read_csv(key) for key in keys]
        df = pd.concat(dfs, ignore_index=True)

        for col in ['entry_date', 'exit_date', 'open_date', 'close_date', 'datetime']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        self.combined_df = df.sort_values(
            ['symbol', 'entry_date']).reset_index(drop=True)
        return self

    def prepare_base_features(self):
        print("\n" + "=" * 70)
        print("STEP 2: PREPARING BASE FEATURES")
        print("=" * 70)

        df = self.combined_df
        df = apply_dynamic_features(df, self.base_formulas)
        self.combined_df = df

        print(" Base features ready\n")
        return self

    def _should_skip_metrics(self):
        if not hasattr(self, "tsfresh_metrics"):
            print("TSFRESH metrics attribute missing — skipping extraction.")
            self.tsfresh_features = pd.DataFrame()
            return True

        if isinstance(self.tsfresh_metrics, dict) and len(self.tsfresh_metrics) == 0:
            print("TSFRESH metrics dict is empty — continuing without extraction.")
            self.tsfresh_features = pd.DataFrame()
            return True

        return False

    def _prepare_sorted_symbol_df(self):
        df = self.combined_df.copy()
        df = df.sort_values(['symbol', 'entry_date']).reset_index(drop=True)
        symbol_df = df[df['symbol'] == self.ticker].sort_values(
            'entry_date').reset_index(drop=True)
        print(f"Processing symbol: {self.ticker}")

        if len(symbol_df) < 11:
            raise ValueError("Not enough trades to extract tsfresh features.")

        return symbol_df

    def _extract_window_features(self, symbol_df, idx, window):
        window_df = symbol_df.iloc[idx - window: idx].reset_index(drop=True)

        ts_data = []

        # convert window into tsfresh long format
        for alias, metric in self.tsfresh_metrics.items():
            if metric not in window_df.columns:
                continue

            for t, v in enumerate(window_df[metric].values):
                ts_data.append({
                    'id': idx,
                    'time': t,
                    'kind': f'{alias}_w{window}',
                    'value': v
                })

        if not ts_data:
            return {}

        ts_df = pd.DataFrame(ts_data)

        # apply tsfresh
        features = extract_features(
            ts_df,
            column_id='id',
            column_sort='time',
            column_kind='kind',
            column_value='value',
            default_fc_parameters=MinimalFCParameters()
        )

        # flatten to dict
        result = {}
        for col in features.columns:
            result[col] = features[col].iloc[0]

        return result

    def _extract_single_trade(self, symbol_df, idx, windows):
        combined_features = {
            'symbol': self.ticker,
            'trade_idx': idx,
            'entry_date': symbol_df.iloc[idx]['entry_date'],
            'exit_date': symbol_df.iloc[idx]['exit_date'],
            'pnl': symbol_df.iloc[idx]['pnl'],
            'label': 1 if symbol_df.iloc[idx]['pnl'] > 0 else 0
        }

        for window in windows:
            if idx >= window:
                fdict = self._extract_window_features(symbol_df, idx, window)
                combined_features.update(fdict)

        return combined_features

    def _finalize_trade_features(self, all_trade_features):
        if not all_trade_features:
            print("WARNING:  No tsfresh features extracted.")
            self.tsfresh_features = pd.DataFrame()
            return

        print(f" Combining {len(all_trade_features)} feature rows...")
        tsfresh_df = pd.DataFrame(all_trade_features)

        # impute only numeric columns
        numeric_cols = tsfresh_df.select_dtypes(include=['float64', 'int64']).columns

        if not numeric_cols.empty:
            tsfresh_numeric = tsfresh_df[numeric_cols]
            tsfresh_numeric = impute(tsfresh_numeric)
            tsfresh_df[numeric_cols] = tsfresh_numeric

        # remove duplicates by symbol+entry_date
        duplicates = tsfresh_df.duplicated(['symbol', 'entry_date']).sum()
        if duplicates > 0:
            print(f"WARNING: Removing {duplicates} duplicate rows")
            tsfresh_df = tsfresh_df.drop_duplicates(['symbol', 'entry_date'])

        self.tsfresh_features = tsfresh_df

    def extract_tsfresh_features_(self):
        if self._should_skip_metrics():
            return self

        symbol_df = self._prepare_sorted_symbol_df()
        if symbol_df is None:
            return self

        all_trade_features = []

        for idx in range(max(self.windows), len(symbol_df)):
            row = self._extract_single_trade(symbol_df, idx, self.windows)
            if row:
                all_trade_features.append(row)

        self._finalize_trade_features(all_trade_features)
        return self

    def save_output(self):
        """Save TSFresh features to Wasabi (CSV + PKL only)"""

        print("\n" + "=" * 70)
        print("STEP 4: SAVING OUTPUT TO WASABI")
        print("=" * 70)

        if self.tsfresh_features is None or len(self.tsfresh_features) == 0:
            print("\n No features to save!")
            return self

        output_df = self.tsfresh_features.sort_values(
            ['symbol', 'entry_date']
        ).reset_index(drop=True)

        # File names based on output path
        base_name = self.ticker + "_" + self.type + '_tsfresh_features' + self.test
        csv_file = base_name + ".csv"
        pkl_file = base_name + ".pkl"

        # Upload to Wasabi
        ws = WasabiStorageSystem(path=f"ml_nonquad/tsfresh_outputs/{self.ticker}", bucket='nufintech-ai-common')

        # print(f" Uploading CSV to Wasabi→ ml_nonquad/tsfresh_outputs/{self.ticker}/{csv_file}")
        print("Uploading CSV to Wasabi")
        ws.write_csv(output_df, symbol="", file=csv_file, index=False)

        print("Uploading Pickle to Wasabi")
        ws.write_pickle(output_df, symbol="", file=pkl_file)

        print("\n Upload complete!")
        print(f"   Rows: {len(output_df):,}")
        print(f"   Columns: {len(output_df.columns)}")
        print("=" * 70)

        return self

    def run(self):
        """Execute  tsfresh extraction pipeline"""

        print("\n" + "=" * 70)
        print("  TSFRESH EXTRACTION (No Duplicates)")
        print("=" * 70)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            self.load_from_wasabi()
            self.prepare_base_features()
            self.extract_tsfresh_features_()
            self.save_output()

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
        description=' tsfresh Feature Extraction (No Duplicates)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    • Processing each trade only once
    • Combining w5 and w10 features into single row
    • No duplicate symbol-dates!
        """
    )

    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Name of the ticker'
    )

    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['long', 'short'],
        help="Whether to load long or short combined file"
    )
    parser.add_argument(
        "--path",
        type=str,
        default='analysis.noquads.atr.251009.10',
        help="the top level path for the keys on Wasabi"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default='.renko_trades.all.v3.4l.top_1.csv',
        help="the suffix for the keys on Wasabi"
    )

    parser.add_argument(
        "--test",
        type=str,
        default='',
        help="Custom tag added to the output filenames for differentiating runs (e.g '' for base, test 1 etc.)."
    )

    args = parser.parse_args()

    # Run  extraction
    extractor = TSFreshExtractor(args.ticker, args.type, args.path, args.suffix, args.test)
    success = extractor.run()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
