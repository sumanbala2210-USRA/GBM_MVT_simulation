import pandas as pd
import os
from datetime import datetime
import numpy as np
import pandas as pd
import numpy as np
import io
from scipy.special import erf # Needed for the Gaussian integral


def get_signal_fraction_in_mvt(row):
    """Calculates the fraction of a pulse's total energy within the MVT window."""
    pulse_shape = row['pulse_shape']
    mvt_s = row['median_mvt_ms'] / 1000.0

    if mvt_s <= 0:
        return 0.0

    if pulse_shape == 'gaussian':
        sigma = row['sigma']#/1000
        # Integral of a Gaussian from -t to +t is proportional to erf(t / (sigma*sqrt(2)))
        # The fraction is the integral over the MVT window divided by the total integral (~1.0)
        z = mvt_s / (2 * sigma * np.sqrt(2))
        return erf(z)

    elif pulse_shape == 'triangular':
        try:
            width = row['width']
            peak_ratio = row['peak_time_ratio']
        except KeyError as e:
            print(f"Missing expected column in row data: {e}")
            return 1.0  # Return full fraction if data is incomplete
        width = row['width']
        peak_ratio = row['peak_time_ratio']
        rise_dur = width * peak_ratio
        decay_dur = width * (1 - peak_ratio)
        
        # We only consider the central part of the MVT window around the peak
        half_mvt = mvt_s / 2.0
        
        # Calculate the area of the small central triangle/trapezoid cut out by the MVT
        area_rise = max(0, (rise_dur**2 - (rise_dur - min(half_mvt, rise_dur))**2)) / rise_dur
        area_decay = max(0, (decay_dur**2 - (decay_dur - min(half_mvt, decay_dur))**2)) / decay_dur

        # The total area of a triangle with height=1 is width/2
        # But since we use peak_amplitude, we normalize by total duration (width)
        return (area_rise + area_decay) / width

    elif pulse_shape in ['norris', 'fred']:
        # The Norris function integral is complex. A reasonable approximation is to
        # assume the MVT is capturing the dominant timescale, which is often the rise time.
        # We can approximate the fraction of flux by the ratio of timescales.
        rise_time = row['rise_time']
        decay_time = row['decay_time']
        # Effective duration is roughly rise + decay
        total_duration_approx = rise_time + decay_time 
        # The fraction is roughly the MVT duration over the total pulse duration
        return min(1.0, mvt_s / total_duration_approx)

    return np.nan # Return NaN if pulse shape is not recognized

def add_snr_mvt(df):
    """Adds SNR_on_mvt and SNR_on_mvt_back columns to the DataFrame."""
    # --- Step 1: Calculate the fraction of signal within the MVT window ---
    df['signal_fraction_in_mvt'] = df.apply(get_signal_fraction_in_mvt, axis=1)

    # --- Step 2 & 3: Estimate Signal (S_mvt) and Background (B_mvt) counts ---
    mvt_duration_s = df['median_mvt_ms'] / 1000.0
    df['S_mvt'] = df['mean_src_counts'] * df['signal_fraction_in_mvt']
    try:
        df['B_mvt'] = df['mean_back_avg_cps'] * mvt_duration_s
    except KeyError as e:
        df['B_mvt'] = df['mean_back_avg'] * mvt_duration_s

    # --- Step 4: Calculate the final SNR ---
    total_counts_in_mvt = df['S_mvt'] + df['B_mvt']
    df['SNR_on_mvt'] = df['S_mvt'] / np.sqrt(total_counts_in_mvt.where(total_counts_in_mvt > 0, np.nan))
    df['SNR_on_mvt_back'] = df['S_mvt'] / np.sqrt(df['B_mvt'].where(df['B_mvt'] > 0, np.nan))
    df["ratio_check"] = df["SNR_on_mvt"] / df["SNR_on_mvt_back"]

    return df


def main():
    now = datetime.now().strftime("%y_%m_%d-%H_%M")

    #file_list = ["norris_combined.csv", "Tringular_all.csv", "LARGE_gauss_combined.csv"]

    file_list = ["99_Gauss_BW.csv"]
    path = os.path.join(os.getcwd(), "01_ANALYSIS_RESULTS")

    for file in file_list:
        file_path = os.path.join(path, file) 

        output_file = os.path.join(path, f"ALL_{file.split('.')[0]}_{now}.csv")


        df = pd.read_csv(file_path)

        # Display the new, relevant columns
        df = add_snr_mvt(df)
        print(df[['median_mvt_ms', 'SNR_on_mvt', 'SNR_on_mvt_back']].head())
        df.to_csv(output_file, index=False)
        # Collect DataFrames
        print(f"CSV saved to:\n{output_file}")

if __name__ == "__main__":
    main()