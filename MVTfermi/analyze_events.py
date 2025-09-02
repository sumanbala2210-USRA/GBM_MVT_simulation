"""
# analyze_events.py
Suman Bala
17 Aug 2025: This script reads a directory of raw TTE event files,
            performs analysis (MVT, SNR) for various bin widths,
            and outputs a final summary CSV file.
"""

# ========= Import necessary libraries =========
import os
import yaml
import logging
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import sys

from SIM_lib import _parse_param, e_n, _create_param_directory_name, send_email, convert_det_to_list, write_yaml


from TTE_SIM_v2 import Function_MVT_analysis, print_nested_dict, check_param_consistency, flatten_dict, GBM_MVT_analysis_det, GBM_MVT_analysis_complex, Function_MVT_analysis_complex



# ========= USER SETTINGS =========
MAX_WORKERS = os.cpu_count() - 2
SIM_CONFIG_FILE = 'simulations_ALL.yaml'
RESULTS_FILE_NAME = "final_summary_results.csv"


                                
def _create_task(base_path, params, settings, bin_width, haar_path, selections=None):
    """Helper function to generate analysis tasks for different detector selections."""
    if selections:
        for selection in selections:
            yield {
                'param_dir_path': base_path,
                'base_params': params,
                'analysis_settings': settings,
                'bin_width_to_process_ms': bin_width,
                'dets_to_analyze': selection,
                'haar_python_path': haar_path
            }
    else:
        yield {
            'param_dir_path': base_path,
            'base_params': params,
            'analysis_settings': settings,
            'bin_width_to_process_ms': bin_width,
            'haar_python_path': haar_path
        }
                


def generate_analysis_tasks(config: Dict[str, Any]) -> 'Generator':
    """Generates all analysis tasks based on the YAML configuration."""
    data_path = Path(config['project_settings']['data_path'])
    haar_python_path = config['project_settings']['haar_python_path']
    analysis_settings = config['analysis_settings']
    bin_widths_to_analyze_ms = analysis_settings.get('bin_widths_to_analyze_ms', [])
    detector_selections = analysis_settings.get('detector_selections')
    pulse_definitions = config['pulse_definitions']

    for campaign in config.get('simulation_campaigns', []):
        if not campaign.get('enabled', False):
            continue 
        sim_type = campaign.get('type')

        for pulse_config in campaign.get('pulses_to_run', []):
            pulse_shape = pulse_config if isinstance(pulse_config, str) else list(pulse_config.keys())[0]
            
            # --- TOP-LEVEL SWITCH for Assembly vs. Standard Mode ---
            if pulse_shape == 'complex_pulse':
                logging.info(f"  > Found '{pulse_shape}', preparing 'assemble-in-analysis' tasks...")
                
                # 1. Manually construct the path to the SINGLE pre-generated template directory.
                #    This is the critical fix: it uses ONLY the fixed canonical parameters.
                template_generation_params = {'peak_amplitude': 1.0, 'position': 0.0}
                if 'trigger_set' in campaign['parameters']:
                    template_generation_params['trigger_set'] = campaign['parameters']['trigger_set'][0]

                template_dir_name = _create_param_directory_name(sim_type, pulse_shape, template_generation_params)
                template_dir_path = data_path / sim_type / pulse_shape / template_dir_name

                if not template_dir_path.exists():
                    logging.error(f"Assembly Error: Template directory for complex_pulse not found at {template_dir_path}")
                    continue

                # 2. Get the full grid of variable parameters for the feature from the YAML.
                base_pulse_config = pulse_definitions.get(pulse_shape, {})
                variable_params_config = campaign['parameters'].copy()
                variable_params_config.update(base_pulse_config.get('parameters', {}))
                
                param_names = list(variable_params_config.keys())
                param_values = [_parse_param(v) for v in variable_params_config.values()]
                total_sim_analysis = campaign['constants'].get('total_sim', 100)

                # 3. Create a task for every combination of the feature parameters.
                for combo in itertools.product(*param_values):
                    current_variable_params = dict(zip(param_names, combo))
                    base_params = {**current_variable_params, 'pulse_shape': pulse_shape, 'sim_type': sim_type, 'num_analysis': total_sim_analysis}
                    
                    # Safety checks for this workflow
                    base_dets = current_variable_params.get('trigger_set', {}).get('det')
                    if base_dets == 'all' and not detector_selections:
                        logging.error(f"Configuration Error: det='all' requires 'detector_selections' in the analysis YAML.")
                        sys.exit(1)
                    if detector_selections and base_dets != 'all' and sim_type == 'gbm':
                        logging.error(f"Configuration Error: 'detector_selections' is active, but simulation is for det='{base_dets}', not 'all'.")
                        sys.exit(1)

                    for bin_width_ms in bin_widths_to_analyze_ms:
                        yield from _create_task(template_dir_path, base_params, analysis_settings, bin_width_ms, haar_python_path, detector_selections)
            
            else: # --- STANDARD MODE for all other simple pulses ---
                base_pulse_config = pulse_definitions.get(pulse_shape, {})
                variable_params_config = campaign['parameters'].copy()
                variable_params_config.update(base_pulse_config.get('parameters', {}))
                total_sim_analysis = campaign['constants'].get('total_sim', 100)
                param_names = list(variable_params_config.keys())
                param_values = [_parse_param(v) for v in variable_params_config.values()]

                for combo in itertools.product(*param_values):
                    current_variable_params = dict(zip(param_names, combo))
                    base_params = {**current_variable_params, 'pulse_shape': pulse_shape, 'sim_type': sim_type, 'num_analysis': total_sim_analysis}
                    
                    # Safety checks for this workflow
                    base_dets = current_variable_params.get('trigger_set', {}).get('det')
                    if base_dets == 'all' and not detector_selections:
                        logging.error(f"Configuration Error: det='all' requires 'detector_selections' in the analysis YAML.")
                        sys.exit(1)
                    if detector_selections and base_dets != 'all' and sim_type == 'gbm':
                        logging.error(f"Configuration Error: 'detector_selections' is active, but simulation is for det='{base_dets}', not 'all'.")
                        sys.exit(1)
                    
                    param_dir_name = _create_param_directory_name(sim_type, pulse_shape, current_variable_params)
                    param_dir_path = data_path / sim_type / pulse_shape / param_dir_name
                    
                    if param_dir_path.exists():
                        for bin_width_ms in bin_widths_to_analyze_ms:
                            yield from _create_task(param_dir_path, base_params, analysis_settings, bin_width_ms, haar_python_path, detector_selections)




# ========= THE CORE WORKER FUNCTION =========
def analyze_one_group(task_info: Dict, data_path: Path, results_path: Path) -> List[Dict]:
    """
    Analyzes one group of NN event files, intelligently handling both 
    'function' (.npz) and 'gbm' (.fits) data products. For each group, it 
    creates a detailed per-realization CSV, a distribution plot, and 
    returns the final summary data.
    """
    # --- 1. Initial Setup ---
    param_dir = task_info['param_dir_path']
    base_params = task_info['base_params']
    analysis_settings = task_info['analysis_settings']
    haar_python_path = task_info['haar_python_path']
    bin_width = task_info['bin_width_to_process_ms']
    sim_type = base_params['sim_type']

    #print("Position:", )

    #print("Base Parameters:---------")
    #print_nested_dict(base_params)
    #exit()
    
    
    #print_nested_dict(base_params)

    # <<< Define standard keys and defaults from your original script >>>
    ALL_PULSE_PARAMS = ['sigma', 'center_time', 'width', 'peak_time_ratio', 'start_time', 'rise_time', 'decay_time']
    DEFAULT_PARAM_VALUE = -999
    STANDARD_KEYS = [
        'sim_type', 'pulse_shape', 'peak_amplitude', 'bin_width_ms',
        'total_sim', 'successful_runs', 'failed_runs',
        'median_mvt_ms', 'mvt_err_lower', 'mvt_err_upper',
        'all_median_mvt_ms', 'all_mvt_err_lower', 'all_mvt_err_upper',
        'angle', 'sim_det', 'base_det', 'analysis_det', 'trigger','background_level',
        'mean_bkgd_counts', 'mean_src_counts', 'mean_back_avg',
        'S_flu', 'S16', 'S32', 'S64', 'S128', 'S256',
        'position'
    ]

    pulse_shape = base_params['pulse_shape']

    ## ================== Directory Creation Logic ==================
    output_analysis_path = None
    if pulse_shape == 'complex_pulse':
        # Assembly Mode: Name the output directory after the variable feature parameters.
        extra_pulse_config = analysis_settings.get('extra_pulse', {})
        feature_pulse_shape = extra_pulse_config.get('pulse_shape', 'extra')
        variable_params_for_name = {
            'peak_amplitude': base_params['peak_amplitude'],
            'position': base_params.get('position', 0.0)
        }
        # Use a placeholder shape like 'gaussian' for consistent naming
        results_dir_name = _create_param_directory_name(sim_type, feature_pulse_shape, variable_params_for_name, extra_pulse=True)
        output_analysis_path = results_path / sim_type / pulse_shape / results_dir_name
    else:
        # Standard Mode: Mirror the input directory structure.
        relative_path = param_dir.relative_to(data_path)
        output_analysis_path = results_path / relative_path

    #print('Path:', output_analysis_path)

    output_analysis_path.mkdir(parents=True, exist_ok=True)
    ## =============================================================
    
    # Create a mirrored output directory in the results path
    #relative_path = param_dir.relative_to(data_path)
    #output_analysis_path = results_path / relative_path
    #output_analysis_path.mkdir(parents=True, exist_ok=True)

    sim_param_file = sorted(param_dir.glob('*.yaml'))
    sim_params = yaml.safe_load(open(sim_param_file[0], 'r'))
    #print("Type:", type(sim_params))
    #print("Simulation Parameters:---------")
    #print_nested_dict(sim_params)
    #print('\n')
    #exit()

    discrepancies = check_param_consistency(
            dict1=sim_params,
            dict2=flatten_dict(base_params),
            pulse_shape=base_params['pulse_shape']
    )
    if discrepancies:
        for msg in discrepancies:
            logging.error(f"Parameter mismatch for {param_dir.name}: {msg}")
        logging.warning(f"Skipping analysis for {param_dir.name} due to parameter mismatch.")
        return [] # Return empty list to skip
    

    det_selection = task_info.get('dets_to_analyze')
    #sim_det = sim_params.get('det')
    #base_det = base_params.get('trigger_set', {}).get('det')
    #analysis_det = convert_det_to_list(det_selection if det_selection and base_det == 'all' else base_det or sim_det)
    #selection_str = "-".join(analysis_det) if isinstance(analysis_det, list) else str(analysis_det or "default")



    #exit(0)
    selection_str = "default"

    analysis_input = {
        'sim_data_path': param_dir,
        'sim_par_file': sim_params,
        'base_params': base_params,
        'analysis_settings': analysis_settings,
        'bin_width_ms': bin_width,
        'haar_python_path': haar_python_path
    }

    # --- 2. Load Data and Perform Per-Realization Analysis ---
    iteration_results, NN = (None, 0)
    if sim_type == 'gbm':
        #det_analysis = convert_det_to_list(det_selection)

        base_dets = base_params['trigger_set']['det']
        if base_dets in ['all', ['all']]:
            analysis_det = convert_det_to_list(det_selection)
        else:
            analysis_det = convert_det_to_list(base_dets)

        #print_nested_dict(base_params)
        if isinstance(analysis_det, list):
            selection_str = "-".join(analysis_det)
        elif analysis_det: # If it's a string like 'n6' or 'all'
            selection_str = str(analysis_det)

        analysis_input['analysis_det'] = analysis_det

        if base_params['pulse_shape'] == 'complex_pulse':
            iteration_results, NN = GBM_MVT_analysis_complex(input_info=analysis_input,
                    output_info = { 'file_path': output_analysis_path,
                                'file_info': param_dir.name})
        else:
            iteration_results, NN = GBM_MVT_analysis_det(input_info=analysis_input,
                    output_info = { 'file_path': output_analysis_path,
                                'file_info': param_dir.name})
        
    else:
        base_dets = analysis_det = sim_params['det']
        if base_params['pulse_shape'] == 'complex_pulse':
            iteration_results, NN = Function_MVT_analysis_complex(input_info=analysis_input,
            output_info={ 'file_path': output_analysis_path,
                         'file_info': param_dir.name})
        else:
         # sim_type == 'function'
            iteration_results, NN = Function_MVT_analysis(input_info=analysis_input,
                output_info={ 'file_path': output_analysis_path,
                            'file_info': param_dir.name})


    # --- 3. Aggregate Results (This logic is common to both data types) ---
    if not iteration_results: return []
    final_summary_list = []

    detailed_df = pd.DataFrame(iteration_results)
    detailed_df.to_csv(output_analysis_path / f"Detailed_{param_dir.name}_{selection_str}.csv", index=False)

    write_yaml(analysis_input, output_analysis_path / f"Params_{param_dir.name}_{selection_str}.yaml")

    # Loop through the DataFrame grouped by the analysis bin width
    valid_runs = detailed_df[detailed_df['mvt_err_ms'] > 0]
    #print(len(valid_runs), "valid runs out of", NN, "for", param_dir.name, "at bin width", bin_width, "ms")
    if len(valid_runs) >= 2:
        # Statistics are calculated ONLY on the valid runs
        p16, median_mvt, p84 = np.percentile(valid_runs['mvt_ms'], [16, 50, 84])
        # Use the 68% confidence interval width as a robust measure of "sigma"
        ci_width = p84 - p16
        # Set plot limits to be wide enough to see the distribution, but not the extreme outliers
        data_min = max(0, p16 - 3 * ci_width)
        data_max = p84 + 10 * ci_width #
        #data_max = np.percentile(all_positive_runs['mvt_ms'], 99.5) if not all_positive_runs.empty else p84 + 3 * ci_width

        # All runs where MVT produced a positive timescale
        all_dist_flag = True
        try:
            all_positive_runs = detailed_df[(detailed_df['mvt_ms'] > 0) & (detailed_df['mvt_ms'] < 1e5)]
            all_p16, all_median_mvt, all_p84 = np.percentile(all_positive_runs['mvt_ms'], [16, 50, 84])
        except:
            all_dist_flag = False
            all_p16, all_median_mvt, all_p84 = (0, 0, 0)

        # --- Create the Enhanced MVT Distribution Plot ---
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # <<< 2. Plot the background histogram of ALL non-failed runs in gray >>>
            if not all_positive_runs.empty and all_dist_flag:
                ax.hist(all_positive_runs['mvt_ms'], bins=30, density=True, 
                            label=f'All Runs w/ MVT > 0 ({len(all_positive_runs)}/{NN})',
                            color='gray', alpha=0.5, histtype='stepfilled', edgecolor='none', zorder=1)
                # Overlay the statistics from the valid runs
                ax.axvline(all_median_mvt, color='k', linestyle='-', lw=1.0,
                        label=f"Median = {all_median_mvt:.3f} ms")
                #ax.axvspan(all_p16, all_p84, color='k', alpha=0.1, hatch='///',
            #label=f"68% C.I. [{all_p16:.3f}, {all_p84:.3f}]")

                ax.axvspan(all_p16, all_p84, color='gray', alpha=0.1,
                        label=f"68% C.I. [{all_p16:.3f}, {all_p84:.3f}]")

            # <<< 3. Plot the main histogram of VALID runs (err > 0) on top >>>
            if len(valid_runs) > 2:
                ax.hist(valid_runs['mvt_ms'], bins=30, density=True, 
                        label=f'Valid Runs w/ Err > 0 ({len(valid_runs)}/{NN})',
                        color='steelblue', histtype='stepfilled', edgecolor='black', zorder=2) 

                # Overlay the statistics from the valid runs
                ax.axvline(median_mvt, color='firebrick', linestyle='-', lw=2.5,
                        label=f"Median = {median_mvt:.3f} ms")
                #ax.axvspan(p16, p84, color='darkorange', alpha=0.3,
                #        label=f"68% C.I. [{p16:.3f}, {p84:.3f}]")
                ax.axvline(p16, color='orange', linestyle='--', lw=1)
                ax.axvline(p84, color='orange', linestyle='--', lw=1)
                ax.axvspan(p16, p84, color='darkorange', alpha=0.1, hatch='///',
                        label=f"68% C.I. [{p16:.3f}, {p84:.3f}]")
                
            auto_min, auto_max = ax.get_xlim()
            final_min = max(auto_min, data_min)
            final_max = min(auto_max, data_max)

            # Formatting
            ax.set_xlim(final_min, final_max)
            ax.set_ylim(bottom=0)
            ax.set_title(f"MVT: {param_dir.name}\nBin Width: {bin_width} ms", fontsize=12)
            ax.set_xlabel("Minimum Variability Timescale (ms)")
            ax.set_ylabel("Probability Density")
            ax.legend()
            fig.tight_layout()
            plt.savefig(output_analysis_path / f"MVT_dis_{param_dir.name}_{selection_str}_{bin_width}ms.png", dpi=300)
            plt.close(fig)
        except Exception as e:
            logging.error(f"Error creating MVT distribution plot for {param_dir.name} at bin width {bin_width}ms: {e}")

        #exit()
        # Build the standardized final output dictionary
        result_data = {
            **base_params, 'bin_width_ms': bin_width,
            'total_sim': NN, 'successful_runs': len(valid_runs),
            'failed_runs': len(detailed_df) - len(valid_runs),
            'median_mvt_ms': round(median_mvt, 4),
            'mvt_err_lower': round(median_mvt - p16, 4),
            'mvt_err_upper': round(p84 - median_mvt, 4),
            'all_median_mvt_ms': round(all_median_mvt, 4),
            'all_mvt_err_lower': round(all_median_mvt - all_p16, 4),
            'all_mvt_err_upper': round(all_p84 - all_median_mvt, 4),
            'mean_src_counts': round(valid_runs['src_counts'].mean(), 2),
            'mean_bkgd_counts': round(valid_runs['bkgd_counts'].mean(), 2),
            'S_flu': round(valid_runs['S_flu'].mean(), 2),
            'S16': round(valid_runs['S16'].mean(), 2),
            'S32': round(valid_runs['S32'].mean(), 2),
            'S64': round(valid_runs['S64'].mean(), 2),
            'S128': round(valid_runs['S128'].mean(), 2),
            'S256': round(valid_runs['S256'].mean(), 2),
            'mean_back_avg': round(valid_runs['back_avg_cps'].mean(), 2),
            'background_level': round(sim_params['background_level'], 2),
            'trigger': sim_params['trigger_number'],
            'sim_det': sim_params['det'],
            'base_det': base_dets,
            'analysis_det': analysis_det,
            'angle': sim_params.get('angle', 0),
            'position': base_params.get('position', 0)
        }
    else:
        logging.warning(f"Creating dummy row for {param_dir.name} at bin width {bin_width}ms due to insufficient valid runs ({len(valid_runs)}).")
        result_data = {
            **base_params, 'bin_width_ms': bin_width,
            'total_sim': NN, 'successful_runs': len(valid_runs),
            'failed_runs': len(detailed_df) - len(valid_runs),
            'median_mvt_ms': -100,
            'mvt_err_lower': -100,
            'mvt_err_upper': -100,
            'all_median_mvt_ms': -100,
            'all_mvt_err_lower': -100,
            'all_mvt_err_upper': -100,
            'mean_src_counts': round(valid_runs['src_counts'].mean(), 2),
            'mean_bkgd_counts': round(valid_runs['bkgd_counts'].mean(), 2),
            'S_flu': round(valid_runs['S_flu'].mean(), 2),
            'S16': round(valid_runs['S16'].mean(), 2),
            'S32': round(valid_runs['S32'].mean(), 2),
            'S64': round(valid_runs['S64'].mean(), 2),
            'S128': round(valid_runs['S128'].mean(), 2),
            'S256': round(valid_runs['S256'].mean(), 2),
            'mean_back_avg': round(valid_runs['back_avg_cps'].mean(), 2),
            'background_level': round(sim_params['background_level'], 2),
            'trigger': sim_params['trigger_number'],
            'sim_det': sim_params['det'],
            'base_det': base_dets,
            'analysis_det': analysis_det,
            'angle': sim_params.get('angle', 0),
            'position': base_params.get('position', 0)
        }

    final_dict = {}
    for key in STANDARD_KEYS:
        final_dict[key] = result_data.get(key)
    
    # Add all possible pulse parameter keys, using the default value if a key is not in this run's result_data
    for key in ALL_PULSE_PARAMS:
        final_dict[key] = result_data.get(key, DEFAULT_PARAM_VALUE)
    
    final_summary_list.append(final_dict)
    return final_summary_list


# ========= MAIN EXECUTION BLOCK (REFACTORED FOR MEMORY STABILITY) =========
def main():
    now = datetime.now().strftime("%y_%m_%d-%H_%M")
    with open(SIM_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
        
    data_path = Path(config['project_settings']['data_path'])
    results_path = Path(config['project_settings']['results_path'])

    analysis_bin = config['analysis_settings']['bin_widths_to_analyze_ms'][0]

    # Create a new, timestamped directory for this run's results
    run_results_path = results_path / f"run_{e_n(analysis_bin)}_{now}"
    run_results_path.mkdir(parents=True, exist_ok=True)
    
    log_file = run_results_path / f'analysis_{now}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    tasks = list(generate_analysis_tasks(config))
    if not tasks:
        logging.warning("No existing simulation directories found to analyze.")
        return

    logging.info(f"Found {len(tasks)} parameter sets to analyze. Starting parallel processing.")
    
    all_results = []
    
    # <<< NEW: Define a chunk size >>>
    # A good starting point is 2-4 times the number of workers.
    CHUNK_SIZE = MAX_WORKERS * 4 

    # <<< NEW: Loop through the tasks in chunks >>>
    for i in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[i:i + CHUNK_SIZE]
        logging.info(f"--- Processing chunk {i//CHUNK_SIZE + 1}/{-(-len(tasks)//CHUNK_SIZE)} (Tasks {i+1}-{i+len(chunk)}) ---")
        
        # <<< The ProcessPoolExecutor is now INSIDE the loop >>>
        # This creates a fresh set of workers for each chunk.
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(analyze_one_group, task, data_path, run_results_path) for task in chunk}
            
            for future in tqdm(as_completed(futures), total=len(chunk), desc="Analyzing Chunk"):
                try:
                    result_list = future.result()
                    if result_list:
                        all_results.extend(result_list)
                except Exception as e:
                    logging.error(f"An analysis task failed in the pool: {e}", exc_info=True)

    # --- Save Final Summary ---
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_results_csv_path = run_results_path / RESULTS_FILE_NAME
        final_df.to_csv(final_results_csv_path, index=False)
        logging.info(f"✅ Analysis complete! Summary saved to:\n{final_results_csv_path}")
    else:
        logging.info("Analysis complete, but no results were generated.")

    #send_email(f"Analysis complete! Summary saved to:\n{final_results_csv_path}")

if __name__ == '__main__':
    main()


"""
# In analyze_one_group, at the very end

    # The 'result_data' dictionary already contains everything we need.
    # We just need to add the generic pulse parameters.
    for key in ALL_PULSE_PARAMS:
        result_data.setdefault(key, DEFAULT_PARAM_VALUE)
    
    # Append the complete dictionary directly to the final list.
    final_summary_list.append(result_data)
    return final_summary_list

"""