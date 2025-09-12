import pandas as pd
import os
from datetime import datetime
import ast

now = datetime.now().strftime("%y_%m_%d-%H_%M")

loc_dist = [
    "complex_gauss_5ms",
    "complex_gauss_10ms",
    "complex_gauss_20ms",
]

path = os.path.join(os.getcwd(), "01_ANALYSIS_RESULTS")

files = [os.path.join(path, f"{loc}/final_summary_results.csv") for loc in loc_dist]

output_file = os.path.join(path, f"combined_{now}.csv")
drop_col_list = ['sim_det', 'analysis_det', 'base_det'] 

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

    # Convert analysis_det strings to lists
    combined_df['analysis_det'] = combined_df['analysis_det'].apply(ast.literal_eval)

    # Add column with number of detections
    combined_df['num_analysis_det'] = combined_df['analysis_det'].apply(len)

    combined_df = combined_df.drop(columns=drop_col_list, errors='ignore')

    cols_all_neg_999 = [col for col in combined_df.columns
                        if combined_df[col].nunique(dropna=True) == 1 and combined_df[col].iloc[0] == -999]
    combined_df = combined_df.drop(columns=cols_all_neg_999)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to:\n{output_file}")
else:
    print("No CSVs found to combine.")
