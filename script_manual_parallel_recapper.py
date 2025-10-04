#!/usr/bin/env python3
"""
Script to analyze JSON result files and compute metrics for different model configurations.
"""

import json
import os
import sys
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

from collections import defaultdict

from analysis import entropy_perf as ep


def load_json_file(filepath):
    """Load JSON file and return data, or None if file doesn't exist or is invalid."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def compute_metrics(data):
    """Compute pass@1, cons@max, avg@max, sequence count, and token count metrics."""
    pass_at_1 = []
    avg_accuracy = []
    cons_at_max = []
    gen_sequences_count = []
    gen_tokens_count = []

    pass_at_1_underM = []
    pass_at_1_sameM = []

    print(f'Found: {len(data)} entries')

    for idx, entry in enumerate(data):
        if str(entry['target']) == 'None':
            # Skip entries with no target
            continue
            
        # count generated tokens
        total_tokens = 0
        for entropy_seq in entry['entropies']:
            tokens_in_seq = sum(1 for entropy in entropy_seq if entropy != -1)  # -2 counts too, only -1 is skipped
            total_tokens += tokens_in_seq
        gen_tokens_count.append(total_tokens)
        
        # count generated sequences
        gen_seqs_entry = len(entry['generations'])
        gen_sequences_count.append(gen_seqs_entry)

        # get target and extracted answers
        target = str(entry['target'])  # Convert to string for comparison
        extracted_answers = entry['extracted_answers']
        
        # pass@1 metric
        # pass_at_1_score = 1 if target in extracted_answers else 0
        pass_at_1_score = 0
        for ele in extracted_answers:
            if target == ele:
                pass_at_1_score = 1
                break
            elif target in ele:
                pass_at_1_score = 1
                break
        pass_at_1.append(pass_at_1_score)

        if gen_seqs_entry < 32:
            pass_at_1_underM.append(pass_at_1_score)
        elif gen_seqs_entry >= 32:
            pass_at_1_sameM.append(pass_at_1_score)     # 32 seqs completate
        
        # average accuracy
        correct_count = sum(1 for answer in extracted_answers if answer == target)
        avg_acc = correct_count / len(extracted_answers) if extracted_answers else 0
        avg_accuracy.append(avg_acc)
        
        # cons@max (majority voting)
        if extracted_answers:
            # count occurrences of each answer
            answer_counts = {}
            for answer in extracted_answers:
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
            # find the answer with maximum count
            max_count = max(answer_counts.values())
            most_frequent_answers = [answer for answer, count in answer_counts.items() if count == max_count]
            
            # If there's a tie, we can choose the first one or handle it differently
            # For now, let's choose the first one in case of tie
            majority_answer = most_frequent_answers[0]
            cons_at_max_score = 1 if majority_answer == target else 0
        else:
            cons_at_max_score = 0
        
        cons_at_max.append(cons_at_max_score)
    
    # Calculate averages
    avg_pass_at_1 = sum(pass_at_1) / len(pass_at_1) if pass_at_1 else 0
    avg_cons_at_max = sum(cons_at_max) / len(cons_at_max) if cons_at_max else 0
    avg_avg_accuracy = sum(avg_accuracy) / len(avg_accuracy) if avg_accuracy else 0
    avg_sequences = sum(gen_sequences_count) / len(gen_sequences_count) if gen_sequences_count else 0
    total_tokens = sum(gen_tokens_count)
    
    return {
        'pass_at_1': avg_pass_at_1,
        'pass_at_1_underM': pass_at_1_underM,
        'pass_at_1_sameM': pass_at_1_sameM,
        'cons_at_max': avg_cons_at_max,
        'avg_at_max': avg_avg_accuracy,
        'avg_sequences': avg_sequences,
        'min_sequences': min(gen_sequences_count) if gen_sequences_count else 0,
        'max_sequences': max(gen_sequences_count) if gen_sequences_count else 0,
        'total_tokens': total_tokens
    }


def find_files(model_name, dataset_name, id_val):
    """Find all possible result files for the given parameters."""
    base_path = f"outputs/{model_name}/{dataset_name}/{id_val}"
    
    files = {}
    
    # Default file
    default_file = f"{base_path}/{id_val}_default_results.large"
    if os.path.exists(default_file):
        files['default'] = default_file
    else:
        # try to find the .json version if .large doesn't exist
        json_default_file = default_file.replace('.large', '.json')
        if os.path.exists(json_default_file):
            # call the parser to convert it to .large
            print(f'-> Converting {json_default_file} to .large format...')
            _ = ep.parse_experiment(
                    exp_name=id_val,
                    model_name=model_name,
                    data_name=dataset_name,
                    metric="mean",
                    to_load="default",
                    percentiles=[99],
                    plot=False,
                    verbose=True,
            )
            if os.path.exists(default_file):
                files['default'] = default_file
            else:
                print(f"Warning: Conversion failed, {default_file} still does not exist.")
    
    # Aware files (with wildcard for threshold)
    aware_pattern = f"{base_path}/{id_val}_aware_results__th*.large"
    aware_files = glob.glob(aware_pattern)
    if aware_files:
        files['aware'] = aware_files[0]  # Take the first match
    else:
        # try to find the .json version if .large doesn't exist
        json_aware_pattern = aware_pattern.replace('.large', '.json')
        json_aware_files = glob.glob(json_aware_pattern)
        if json_aware_files:
            print(f'-> Converting {json_aware_files[0]} to .large format...')
            _ = ep.parse_experiment(
                    exp_name=id_val,
                    model_name=model_name,
                    data_name=dataset_name,
                    metric="mean",
                    to_load="entropy",
                    percentiles=[99],
                    plot=False,
                    verbose=True,
            )
            aware_large_files = glob.glob(aware_pattern)
            if aware_large_files:
                files['aware'] = aware_large_files[0]
            else:
                print(f"Warning: Conversion failed, no aware .large file found.")
    
    # More budget files (with wildcard for threshold)
    more_pattern = f"{base_path}/{id_val}_aware_more_budget_results__th*.large"
    more_files = glob.glob(more_pattern)
    if more_files:
        files['more'] = more_files[0]  # Take the first match
    else:
        # try to find the .json version if .large doesn't exist
        json_more_pattern = more_pattern.replace('.large', '.json')
        json_more_files = glob.glob(json_more_pattern)
        if json_more_files:
            print(f'-> Converting {json_more_files[0]} to .large format...')
            _ = ep.parse_experiment(
                    exp_name=id_val,
                    model_name=model_name,
                    data_name=dataset_name,
                    metric="mean",
                    to_load="more",
                    percentiles=[99],
                    plot=False,
                    verbose=True,
            )
            more_large_files = glob.glob(more_pattern)
            if more_large_files:
                files['more'] = more_large_files[0]
            else:
                print(f"Warning: Conversion failed, no more budget .large file found.")

    more_easy_pattern = f"{base_path}/{id_val}_aware_easy_more_budget_results__th*.large"
    more_easy_files = glob.glob(more_easy_pattern)
    if more_easy_files:
        files['more_easy'] = more_easy_files[0]  # Take the first match
    else:
        # try to find the .json version if .large doesn't exist
        json_more_easy_pattern = more_easy_pattern.replace('.large', '.json')
        json_more_easy_files = glob.glob(json_more_easy_pattern)
        if json_more_easy_files:
            print(f'-> Converting {json_more_easy_files[0]} to .large format...')
            _ = ep.parse_experiment(
                    exp_name=id_val,
                    model_name=model_name,
                    data_name=dataset_name,
                    metric="mean",
                    to_load="more_easy",
                    percentiles=[99],
                    plot=False,
                    verbose=True,
            )
            more_easy_large_files = glob.glob(more_easy_pattern)
            if more_easy_large_files:
                files['more_easy'] = more_easy_large_files[0]
            else:
                print(f"Warning: Conversion failed, no more easy budget .large file found.")


    return files


def find_files_wrapper(params):
    """Wrapper function for parallel execution of find_files."""
    model_name, dataset_name, id_val = params
    try:
        files = find_files(model_name, dataset_name, id_val)
        return (model_name, dataset_name, id_val, files, None)
    except Exception as e:
        return (model_name, dataset_name, id_val, None, str(e))


def run_parallel_find_files(params_list, max_workers=None):
    """Run find_files function in parallel for all parameter combinations."""
    if max_workers is None:
        max_workers = min(len(params_list), multiprocessing.cpu_count())
    
    print(f"Running {len(params_list)} jobs in parallel with {max_workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(find_files_wrapper, params): params 
                           for params in params_list}
        
        # Process completed jobs with progress bar
        with tqdm(total=len(params_list), desc="Processing files") as pbar:
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update progress bar with current job info
                    model_name, dataset_name, id_val = params
                    pbar.set_postfix_str(f"{model_name[:8]}|{dataset_name[:8]}|{id_val[-8:]}")
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError processing {params}: {e}")
                    results.append((params[0], params[1], params[2], None, str(e)))
                    pbar.update(1)
    
    # Print summary
    print("\n" + "="*80)
    print("PARALLEL EXECUTION SUMMARY")
    print("="*80)
    
    successful_jobs = 0
    failed_jobs = 0
    
    for model_name, dataset_name, id_val, files, error in results:
        if error is None:
            successful_jobs += 1
            file_count = len(files) if files else 0
            print(f"✓ {model_name} | {dataset_name} | {id_val} -> {file_count} files found")
        else:
            failed_jobs += 1
            print(f"✗ {model_name} | {dataset_name} | {id_val} -> Error: {error}")
    
    print(f"\nCompleted: {successful_jobs} successful, {failed_jobs} failed")
    return results


def analyze_results(model_name, dataset_name, id_val):
    """Main analysis function."""
    files = find_files(model_name, dataset_name, id_val)
    print("\n\n\n" + str(files))

    if not files:
        print(f"No result files found for {model_name} | {dataset_name} | {id_val}")
        return
    
    for file_type, filepath in files.items():
        print(f"\n{model_name} | {dataset_name} | {id_val} | {file_type}")

        if file_type.lower() != 'default':
            print(f"- threshold: {filepath.split('__th')[-1].replace('.large','')}")
        
        data = load_json_file(filepath)
        if data is None:
            continue
        try:
            metrics = compute_metrics(data)

            avg_pass_at_1_underM = sum(metrics['pass_at_1_underM']) / len(metrics['pass_at_1_underM']) if metrics['pass_at_1_underM'] else 0
            avg_pass_at_1_sameM = sum(metrics['pass_at_1_sameM']) / len(metrics['pass_at_1_sameM']) if metrics['pass_at_1_sameM'] else 0

            print(f"- pass@1: {metrics['pass_at_1']:.2f}")
            print(f"\t\t M=32 ({len(metrics['pass_at_1_sameM'])} seqs): their pass@1: {avg_pass_at_1_sameM:.2f}")
            print(f"\t\t M<32 ({len(metrics['pass_at_1_underM'])} seqs): thier pass@1: {avg_pass_at_1_underM:.2f}")
            print(f"- cons@max: {metrics['cons_at_max']:.2f} (majority voting)")
            print(f"- avg@max: {metrics['avg_at_max']:.2f} (average accuracy)")
            print(f"- # token: {metrics['total_tokens']} sum of all tokens ({metrics['total_tokens'] / 1000000:.0f}M average)")
            print(f"- # seq: {metrics['avg_sequences']:.1f} average sequence count (min: {metrics['min_sequences']}, max: {metrics['max_sequences']})")
            
        except Exception as e:
            print(f"Error computing metrics for {filepath}: {e}")





######################################## FOR table_latex

def get_threshold_from_filename(filename):
    """Extract threshold value from filename"""
    if '__th' in filename:
        threshold_part = filename.split('__th')[-1].replace('.large', '').replace('.json', '')
        try:
            return float(threshold_part)
        except ValueError:
            return None
    return None

def categorize_method(filename, threshold):
    """Determine if this is entropy or more method based on filename patterns"""
    # You'll need to adjust this based on your actual filename patterns
    if 'entropy' in filename.lower():
        return 'entropy'
    elif 'more' in filename.lower():
        return 'more'
    elif 'easy_more' in filename.lower():
        return 'more_easy'
    elif threshold is not None:
        # If you can't tell from filename, you might need additional logic here
        return 'entropy'  # or 'more', depending on your convention
    return 'default'

def collect_all_metrics(experiments):
    """Collect all metrics for every experiment into a comprehensive data structure"""
    
    # Main data structure: {model: {dataset: {threshold_key: metrics}}}
    all_metrics = defaultdict(lambda: defaultdict(dict))
    
    for model_name, dataset_name, id_val in tqdm(experiments, desc="Collecting metrics"):
        files = find_files(model_name, dataset_name, id_val)
        
        if not files:
            print(f"No files found for {model_name} | {dataset_name} | {id_val}")
            continue
            
        for file_type, filepath in files.items():
            data = load_json_file(filepath)
            if data is None:
                continue
                
            try:
                metrics = compute_metrics(data)
                threshold = get_threshold_from_filename(filepath) if file_type != 'default' else None
                method = categorize_method(filepath, threshold)
                
                # Create a comprehensive metrics entry
                metric_entry = {
                    'pass_at_1': metrics['pass_at_1'],
                    'cons_at_max': metrics['cons_at_max'],
                    'avg_at_max': metrics['avg_at_max'],
                    'avg_sequences': metrics['avg_sequences'],
                    'min_sequences': metrics['min_sequences'],
                    'max_sequences': metrics['max_sequences'],
                    'avg_tokens': metrics['total_tokens'] / len(data) if len(data) > 0 else 0,  # Average tokens per problem
                    'total_tokens': metrics['total_tokens'],
                    'method': method,
                    'threshold': threshold,
                    'filepath': filepath,
                    'exp_id': id_val  # Store experiment ID within metrics
                }
                
                # Store using file_type and threshold as key
                if threshold is not None:
                    threshold_key = f"{file_type}_{threshold}"
                else:
                    threshold_key = "default"
                    
                all_metrics[model_name][dataset_name][threshold_key] = metric_entry
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
    
    return dict(all_metrics)

def save_metrics_to_json(all_metrics, filename="experiment_metrics.json"):
    """Save all metrics to a JSON file"""
    
    # Convert defaultdict to regular dict for JSON serialization
    def convert_defaultdict(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        return obj
    
    # Deep convert all nested defaultdicts
    def deep_convert(obj):
        if isinstance(obj, defaultdict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        else:
            return obj
    
    serializable_metrics = deep_convert(all_metrics)
    
    with open(filename, 'w') as f:
        json.dump(serializable_metrics, f, indent=4, default=str, ensure_ascii=False)
    
    print(f"All metrics saved to {filename}")
    return serializable_metrics

def print_metrics_summary(all_metrics):
    """Print a summary of collected metrics"""
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    
    for model_name in sorted(all_metrics.keys()):
        print(f"\n{model_name}:")
        print("-" * 40)
        
        for dataset_name in sorted(all_metrics[model_name].keys()):
            print(f"  Dataset: {dataset_name}")
            
            thresholds = sorted(all_metrics[model_name][dataset_name].keys())
            for threshold_key in thresholds:
                metrics = all_metrics[model_name][dataset_name][threshold_key]
                
                print(f"    {threshold_key} (exp_id: {metrics['exp_id']}):")
                print(f"      pass@1: {metrics['pass_at_1']:.3f}")
                print(f"      cons@max: {metrics['cons_at_max']:.3f}")
                print(f"      avg@max: {metrics['avg_at_max']:.3f}")
                print(f"      avg_tokens: {metrics['avg_tokens']:.0f}")
                print(f"      total_tokens: {metrics['total_tokens']}")
                print(f"      sequences: {metrics['avg_sequences']:.1f} (min: {metrics['min_sequences']}, max: {metrics['max_sequences']})")
                print(f"      method: {metrics['method']}")


def main():
    parser = argparse.ArgumentParser(description='Analyze JSON result files and compute metrics')
    parser.add_argument('model_name', nargs='?', help='Model name (not used in parallel mode)')
    parser.add_argument('dataset_name', nargs='?', help='Dataset name (not used in parallel mode)')
    parser.add_argument('id', nargs='?', help='ID value (not used in parallel mode)')
    parser.add_argument('--parallel', action='store_true', 
                       help='Run in parallel mode using predefined parameter list')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--recap', action='store_true',
                        help='Collect and summarize all metrics from predefined experiments')
    
    args = parser.parse_args()
    
    if args.parallel:
        run_parallel_find_files(params, max_workers=args.workers)
        print("\nParallel execution completed. No analysis performed.")

    elif args.recap:
        print("Collecting all metrics...")
        all_metrics = collect_all_metrics(params)

        saved_metrics = save_metrics_to_json(all_metrics, "experiment_metrics.json")
        # Print summary
        print_metrics_summary(all_metrics)

    else:
        # Original single-job mode
        if not all([args.model_name, args.dataset_name, args.id]):
            parser.error("model_name, dataset_name, and id are required when not using --parallel")
        
        analyze_results(args.model_name, args.dataset_name, args.id)




params = [
    ["SmolLM3-3B", "AIME_2024", "2025-07-14_12-38-33"],  # default
    ["SmolLM3-3B", "AIME_2024", "2025-07-27_17-34-09"],
    ["SmolLM3-3B", "AIME_2024", "2025-09-11_00-13-29"],
    ["SmolLM3-3B", "AIME_2024", "2025-07-27_17-26-06"],
    ["SmolLM3-3B", "AIME_2024", "2025-09-11_02-03-42"],
    ["SmolLM3-3B", "AIME_2024", "2025-07-14_13-43-27"],
    ["SmolLM3-3B", "AIME_2024", "2025-08-09_20-57-18"],

    ["SmolLM3-3B", "AIME2025", "2025-07-15_19-30-34"],  # default
    ["SmolLM3-3B", "AIME2025", "2025-08-09_20-53-18"],
    ["SmolLM3-3B", "AIME2025", "2025-07-15_19-30-34"],
    ["SmolLM3-3B", "AIME2025", "2025-07-27_06-13-12"],
    ["SmolLM3-3B", "AIME2025", "2025-07-27_06-13-11"],
    ["SmolLM3-3B", "AIME2025", "2025-07-27_06-59-14"],
    ["SmolLM3-3B", "AIME2025", "2025-07-27_12-07-45"],

    ["SmolLM3-3B", "hmmt_feb_2025", "2025-09-05_06-30-31"],  # default
    ["SmolLM3-3B", "hmmt_feb_2025", "2025-09-05_11-42-28"],
    ["SmolLM3-3B", "hmmt_feb_2025", "2025-09-05_10-37-42"],
    ["SmolLM3-3B", "hmmt_feb_2025", "2025-09-05_08-17-01"],
    ["SmolLM3-3B", "hmmt_feb_2025", "2025-09-05_07-36-50"],
    ["SmolLM3-3B", "hmmt_feb_2025", "2025-09-05_07-30-49"],
    ["SmolLM3-3B", "hmmt_feb_2025", "2025-09-05_07-22-47"],
    ["SmolLM3-3B", "hmmt_feb_2025", "2025-09-05_07-22-46"],

    ["SmolLM3-3B", "GPQA-Diamond", "2025-08-13_11-20-05"],  # default
    ["SmolLM3-3B", "GPQA-Diamond", "2025-08-20_08-16-45"],
    ["SmolLM3-3B", "GPQA-Diamond", "2025-08-20_06-52-42"],
    ["SmolLM3-3B", "GPQA-Diamond", "2025-08-13_12-40-12"],
    ["SmolLM3-3B", "GPQA-Diamond", "2025-08-13_13-34-12"],
    ["SmolLM3-3B", "GPQA-Diamond", "2025-08-13_13-34-13"],
    ["SmolLM3-3B", "GPQA-Diamond", "2025-08-13_13-34-14"],

    ["SmolLM3-3B", "humanevalplus", "2025-09-11_07-16-07"],  # default
    # ["SmolLM3-3B", "humanevalplus", "2025-09-18_21-59-40"],   # results more coming (habrok)
    # ["SmolLM3-3B", "humanevalplus", "2025-09-18_20-46-50"],   # results more coming (habrok)
    # ["SmolLM3-3B", "humanevalplus", "2025-09-18_20-17-04"],   # results more coming (habrok)
    ["SmolLM3-3B", "humanevalplus", "2025-09-18_20-17-05"],   # results more coming (habrok)
    ["SmolLM3-3B", "humanevalplus", "2025-09-18_20-17-06"],   # results more coming (habrok)
    ["SmolLM3-3B", "humanevalplus", "2025-09-18_20-17-07"],   # missing more (not doin)

    ["Qwen3-4B", "AIME_2024", "2025-07-17_12-47-34"],  # default
    ["Qwen3-4B", "AIME_2024", "2025-08-13_11-15-57"],
    ["Qwen3-4B", "AIME_2024", "2025-07-27_17-52-09"],
    ["Qwen3-4B", "AIME_2024", "2025-09-11_02-49-46"],
    ["Qwen3-4B", "AIME_2024", "2025-07-27_17-42-08"],
    ["Qwen3-4B", "AIME_2024", "2025-09-11_03-49-51"],
    ["Qwen3-4B", "AIME_2024", "2025-07-17_12-47-34"],

    ["Qwen3-4B", "AIME2025", "2025-07-17_13-55-50"],  # default
    ["Qwen3-4B", "AIME2025", "2025-08-09_15-29-12"],
    ["Qwen3-4B", "AIME2025", "2025-08-09_13-59-06"],
    ["Qwen3-4B", "AIME2025", "2025-07-31_04-23-45"],
    ["Qwen3-4B", "AIME2025", "2025-07-31_04-11-44"],
    ["Qwen3-4B", "AIME2025", "2025-07-27_17-08-04"],
    ["Qwen3-4B", "AIME2025", "2025-07-27_16-36-01"],
    ["Qwen3-4B", "AIME2025", "2025-07-18_21-24-31"],

    ["Qwen3-4B", "hmmt_feb_2025", "2025-08-31_18-39-25"],  # default
    ["Qwen3-4B", "hmmt_feb_2025", "2025-09-05_20-56-55"],
    ["Qwen3-4B", "hmmt_feb_2025", "2025-09-05_19-32-35"],
    ["Qwen3-4B", "hmmt_feb_2025", "2025-09-05_18-34-21"],
    ["Qwen3-4B", "hmmt_feb_2025", "2025-09-05_18-12-16"],
    ["Qwen3-4B", "hmmt_feb_2025", "2025-09-05_17-58-13"],
    ["Qwen3-4B", "hmmt_feb_2025", "2025-09-05_17-54-12"],
    ["Qwen3-4B", "hmmt_feb_2025", "2025-09-05_11-48-15"],

    ["Qwen3-4B", "GPQA-Diamond", "2025-08-13_12-40-11"],  # default
    ["Qwen3-4B", "GPQA-Diamond", "2025-08-20_08-30-48"],
    ["Qwen3-4B", "GPQA-Diamond", "2025-08-13_23-18-35"],
    ["Qwen3-4B", "GPQA-Diamond", "2025-08-13_22-56-34"],
    ["Qwen3-4B", "GPQA-Diamond", "2025-08-13_18-52-24"],
    ["Qwen3-4B", "GPQA-Diamond", "2025-08-13_17-42-24"],
    ["Qwen3-4B", "GPQA-Diamond", "2025-08-13_13-50-12"],

    ["Qwen3-4B", "humanevalplus", "2025-09-11_08-18-12"],  # default
    ["Qwen3-4B", "humanevalplus", "2025-09-18_20-17-07"],
    ["Qwen3-4B", "humanevalplus", "2025-09-18_20-17-05"],
    ["Qwen3-4B", "humanevalplus", "2025-09-18_20-17-06"],
    ["Qwen3-4B", "humanevalplus", "2025-09-18_19-51-36"],
    ["Qwen3-4B", "humanevalplus", "2025-09-18_14-49-47"],
    ["Qwen3-4B", "humanevalplus", "2025-09-18_14-49-48"],
    ["Qwen3-4B", "humanevalplus", "2025-09-12_06-56-53"],

    ["DeepSeek-R1-0528-Qwen3-8B", "AIME_2024", "2025-07-17_14-48-35"],  # default
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME_2024", "2025-07-09_11-31-25"],
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME_2024", "2025-09-11_04-25-54"],
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME_2024", "2025-07-27_17-26-07"],
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME_2024", "2025-09-11_04-47-57"],
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME_2024", "2025-07-10_19-05-25"],

    ["DeepSeek-R1-0528-Qwen3-8B", "AIME2025", "2025-07-19_05-35-01"],  # default
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME2025", "2025-07-19_06-01-03"],
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME2025", "2025-07-31_09-20-34"],
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME2025", "2025-08-06_02-13-42"],
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME2025", "2025-08-09_10-23-01"],
    ["DeepSeek-R1-0528-Qwen3-8B", "AIME2025", "2025-08-07_02-43-31"],

    ["DeepSeek-R1-0528-Qwen3-8B", "hmmt_feb_2025", "2025-08-31_18-43-25"],  # default
    ["DeepSeek-R1-0528-Qwen3-8B", "hmmt_feb_2025", "2025-09-01_19-49-05"],
    ["DeepSeek-R1-0528-Qwen3-8B", "hmmt_feb_2025", "2025-09-01_19-49-06"],
    ["DeepSeek-R1-0528-Qwen3-8B", "hmmt_feb_2025", "2025-09-01_19-23-02"],
    ["DeepSeek-R1-0528-Qwen3-8B", "hmmt_feb_2025", "2025-09-01_18-52-59"],
    ["DeepSeek-R1-0528-Qwen3-8B", "hmmt_feb_2025", "2025-09-01_18-46-58"],
    ["DeepSeek-R1-0528-Qwen3-8B", "hmmt_feb_2025", "2025-09-01_18-22-52"],
    ["DeepSeek-R1-0528-Qwen3-8B", "hmmt_feb_2025", "2025-08-31_23-29-42"],

    ["DeepSeek-R1-0528-Qwen3-8B", "hmmt_feb_2025", "2025-09-01_19-49-06"],  # easy_more

    ["DeepSeek-R1-0528-Qwen3-8B", "GPQA-Diamond", "2025-09-20_11-55-02"],  # default
    ["DeepSeek-R1-0528-Qwen3-8B", "GPQA-Diamond", "2025-09-18_22-33-54"],
    ["DeepSeek-R1-0528-Qwen3-8B", "GPQA-Diamond", "2025-09-19_01-56-44"],
    ["DeepSeek-R1-0528-Qwen3-8B", "GPQA-Diamond", "2025-09-18_23-53-29"],     # more to integrate manually (from habrok)
    # ["DeepSeek-R1-0528-Qwen3-8B", "GPQA-Diamond", "2025-09-18_22-55-45"],     # more to integrate manually (from habrok)
    ["DeepSeek-R1-0528-Qwen3-8B", "GPQA-Diamond", "2025-09-18_22-45-54"],     # more to integrate manually (from habrok)

    ["DeepSeek-R1-0528-Qwen3-8B", "humanevalplus", "2025-09-11_21-17-48"],  # default
    ["DeepSeek-R1-0528-Qwen3-8B", "humanevalplus", "2025-09-12_06-06-49"],
    ["DeepSeek-R1-0528-Qwen3-8B", "humanevalplus", "2025-09-12_06-02-49"],
    ["DeepSeek-R1-0528-Qwen3-8B", "humanevalplus", "2025-09-12_05-00-43"],
    ["DeepSeek-R1-0528-Qwen3-8B", "humanevalplus", "2025-09-12_04-48-41"],
    ["DeepSeek-R1-0528-Qwen3-8B", "humanevalplus", "2025-09-12_04-42-41"],
    ["DeepSeek-R1-0528-Qwen3-8B", "humanevalplus", "2025-09-12_04-12-36"],
    ["DeepSeek-R1-0528-Qwen3-8B", "humanevalplus", "2025-09-12_02-26-18"],      # missing more (not doin)

    ["gpt-oss-20b", "AIME_2024", "2025-09-16_18-31-12"],  # default
    ["gpt-oss-20b", "AIME_2024", "2025-09-16_23-29-33"],
    ["gpt-oss-20b", "AIME_2024", "2025-09-17_05-43-15"],
    ["gpt-oss-20b", "AIME_2024", "2025-09-17_12-19-24"],
    ["gpt-oss-20b", "AIME_2024", "2025-09-17_18-49-17"],
    ["gpt-oss-20b", "AIME_2024", "2025-09-18_01-49-26"],
    ["gpt-oss-20b", "AIME_2024", "2025-09-18_08-59-28"],

    ["gpt-oss-20b", "AIME2025", "2025-09-15_10-37-01"],  # default
    ["gpt-oss-20b", "AIME2025", "2025-09-15_17-49-57"],
    ["gpt-oss-20b", "AIME2025", "2025-09-15_23-50-22"],
    ["gpt-oss-20b", "AIME2025", "2025-09-16_05-49-11"],
    ["gpt-oss-20b", "AIME2025", "2025-09-16_11-18-53"],
    ["gpt-oss-20b", "AIME2025", "2025-09-16_18-08-23"],
    ["gpt-oss-20b", "AIME2025", "2025-09-19_00-09-21"],

    ["gpt-oss-20b", "hmmt_feb_2025", "2025-09-16_18-08-23"],  # default
    ["gpt-oss-20b", "hmmt_feb_2025", "2025-09-17_05-20-28"],
    ["gpt-oss-20b", "hmmt_feb_2025", "2025-09-17_10-42-19"],
    ["gpt-oss-20b", "hmmt_feb_2025", "2025-09-17_16-07-33"],
    ["gpt-oss-20b", "hmmt_feb_2025", "2025-09-17_22-13-30"],
    ["gpt-oss-20b", "hmmt_feb_2025", "2025-09-18_04-53-24"],
    ["gpt-oss-20b", "hmmt_feb_2025", "2025-09-18_12-07-05"],

    ["gpt-oss-20b", "GPQA-Diamond", "2025-09-19_09-49-41"],  # default
    ["gpt-oss-20b", "GPQA-Diamond", "2025-09-19_20-23-45"],
    ["gpt-oss-20b", "GPQA-Diamond", "2025-09-20_09-12-42"],
    ["gpt-oss-20b", "GPQA-Diamond", "2025-09-20_22-12-12"],
    ["gpt-oss-20b", "GPQA-Diamond", "2025-09-21_10-48-04"],
    ["gpt-oss-20b", "GPQA-Diamond", "2025-09-22_00-35-58"],
    
    ["gpt-oss-20b", "humanevalplus", "2025-09-18_18-34-30"],  # default
    ["gpt-oss-20b", "humanevalplus", "2025-09-18_20-44-50"],
    ["gpt-oss-20b", "humanevalplus", "2025-09-20_15-18-34"],
    ["gpt-oss-20b", "humanevalplus", "2025-09-21_05-40-04"],
    ["gpt-oss-20b", "humanevalplus", "2025-09-21_18-58-01"],
    # ["gpt-oss-20b", "humanevalplus", "2025-09-22_07-26-19"],  # results more coming (colossus)
    # ["gpt-oss-20b", "humanevalplus", ""],
]








if __name__ == "__main__":
    main()
