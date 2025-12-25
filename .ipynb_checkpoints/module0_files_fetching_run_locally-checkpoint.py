# import pandas as pd
import os
import glob
import shutil

main_folder = r'.\analysis.noquads.atr.251009.10'
dest_folder = r'.\selected_files'

long_pattern = "*_long.renko_trades.all.v3.4l.top_1.csv"
short_pattern = "*_short.renko_trades.all.v3.4l.top_1.csv"

os.makedirs(dest_folder, exist_ok=True)

symbol_dirs = [
    os.path.join(main_folder, d)
    for d in os.listdir(main_folder)
    if os.path.isdir(os.path.join(main_folder, d))
]

for sym_dir in symbol_dirs:
    long_files = glob.glob(os.path.join(sym_dir, long_pattern))
    short_files = glob.glob(os.path.join(sym_dir, short_pattern))

    for f in long_files + short_files:
        try:
            shutil.copy2(f, dest_folder)
            print(f"Copied: {f}")
        except Exception as e:
            print(f"Error copying {f}: {e}")

print(" Copy completed.")
