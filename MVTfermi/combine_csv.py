import pandas as pd
import os
from datetime import datetime

now = datetime.now().strftime("%y_%m_%d-%H_%M")

loc_dist = [
    "complex_gauss_1ms",
    "run_1_25_09_03-19_05",
]

path = os.path.join(os.getcwd(), "01_ANALYSIS_RESULTS")

files = [os.path.join(path, f"{loc}/final_summary_results.csv") for loc in loc_dist]

output_file = os.path.join(path, f"combined_{now}.csv")

# Collect DataFrames
dfs = []
for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        if "sigma" in df.columns:
            df["sigma"] = df["sigma"] * 1#000
        dfs.append(df)
    else:
        print(f"Warning: File not found -> {f}")

# Combine them
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to:\n{output_file}")
else:
    print("No CSVs found to combine.")
