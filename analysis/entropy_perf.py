import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import ast
import os
from pathlib import Path
from tqdm import tqdm

def parse_string_to_list(entropy_str):
    """Parse entropy string to list of floats, handling various formats"""
    try:
        values = ast.literal_eval(entropy_str)
        return values
    except:
        return []

def calculate_entropy_peaks(entropies, percentile=99):
    """Calculate high entropy peaks (e.g., 99th percentile) for each sequence"""
    peaks = []
    for entropy_list in entropies:
        if entropy_list:
            peak = np.percentile(entropy_list, percentile)
            peaks.append(peak)
        else:
            peaks.append(0)  # Default if parsing fails
    
    return peaks

def calculate_accuracy(target, extracted_answers):
    """Calculate accuracy as the fraction of correct answers"""
    if not extracted_answers:
        return 0.0
    
    target_str = str(target)

    correct_count = sum(1 for answer in extracted_answers if str(answer) == target_str)

    return correct_count / len(extracted_answers)

def analyze_entropy_accuracy_correlation(
    data,
    entropy_percentile=99,
    metric_to_use=np.median,
    select_less_sequences=None,
):
    """
    Analyze correlation between entropy peaks and accuracy
    
    Args:
        data: List of dictionaries with target, extracted_answers, and entropies
        entropy_percentile: Percentile to use for entropy peaks (default 99)
    
    Returns:
        Dictionary with analysis results
    """
    entropy_measures = []
    accuracies = []
    thresholds = []
    
    for item in data:
        if select_less_sequences is None:
            indexes_to_select = range(0, len(item['entropies']))      # all sequences in
        else:
            random_indexes = np.random.choice(len(item['entropies']), size=select_less_sequences, replace=False)
            indexes_to_select = sorted(random_indexes)                 # sorted to keep the original order
        target = item['target']
        extracted_answers = [item['extracted_answers'][i] for i in indexes_to_select]
        entropies = [item['entropies'][i] for i in indexes_to_select]
        
        # Calculate accuracy
        accuracy = calculate_accuracy(target, extracted_answers)
        
        # Calculate entropy peaks
        peak_thresholds = calculate_entropy_peaks(entropies, entropy_percentile)  # list of 99th percentile peaks, len(gen sequences)  [2.1, 2.3, 1.9, ...]
        thresholds.extend(peak_thresholds)

        # filter values in entropies higher than the threshold
        # for each sequence compute the 99th percentile, take all the entropies that are higher than the 99th percentile and consider them as peaks.
        # note the 99th percentile is sequence dependent, meaning that for each prompt and each generated sequence the 99th percentile may be different.
        entropy_peaks = []
        for i, th in enumerate(peak_thresholds):
            peaks = [e for e in entropies[i] if e >= th]
            assert len(peaks) > 0, f"Error: No entropies found above current threshold {th} for item with target: {target}, sequence index: {i}"        # this should not happen

            entropy_peaks.extend(peaks)
        
        # Use mean/median of entropy peaks as the entropy measure for this item
        if entropy_peaks:
            entropy_measure = metric_to_use(entropy_peaks)          # metric to use can be np.mean, np.median, np.max, etc.
            entropy_measures.append(entropy_measure)
            accuracies.append(accuracy)
    
    # Calculate correlations between the chosen metric and the accuracy
    if len(entropy_measures) > 1:
        pearson_corr, pearson_p = pearsonr(entropy_measures, accuracies)
        spearman_corr, spearman_p = spearmanr(entropy_measures, accuracies)
    else:
        print("Not enough data points to calculate correlations.")
        pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
    
    return {
        'entropy_measures': entropy_measures,   # list of mean/median entropy peaks
        'accuracies': accuracies,               # list of accuracies (fraction correct)
        'pearson_correlation': pearson_corr,    # Pearson correlation coefficient
        'pearson_p_value': pearson_p,           # p-value for Pearson correlation
        'spearman_correlation': spearman_corr,  # Spearman correlation coefficient
        'spearman_p_value': spearman_p,         # p-value for Spearman correlation
        'n_samples': len(entropy_measures),     # number of valid samples
        'thresholds': thresholds                # list of all 99th percentile thresholds used
    }

def plot_entropy_accuracy_scatter(
    results,
    entropy_percentile=99,
    experiment_path=None,
    aggregation_method="mean",
):
    """Create scatter plot of entropy vs accuracy"""
    plt.figure(figsize=(6, 4))
    
    entropy_measures = results['entropy_measures']
    accuracies = results['accuracies']
    
    plt.scatter(entropy_measures, accuracies, alpha=0.6, s=50)
    
    # Add trend line
    if len(entropy_measures) > 1:
        z = np.polyfit(entropy_measures, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(entropy_measures, p(entropy_measures), "r--", alpha=0.8)
    
    plt.xlabel(f'{aggregation_method} {entropy_percentile}th Percentile Entropy')
    plt.ylabel('Accuracy (Fraction correct)')
    plt.title('Entropy Peaks vs Generation Accuracy')
    plt.grid(True, alpha=0.2)
    
    plt.ylim(-0.09, 1.09)  # Accuracy is between 0 and 1

    # Add correlation info to plot
    pearson_corr = results['pearson_correlation']
    if not np.isnan(pearson_corr):
        plt.text(0.05, 0.2, f'Pearson r = {pearson_corr:.2f} (p={results["pearson_p_value"]:.3f})\nSpearman r = {results["spearman_correlation"]:.2f} (p={results["spearman_p_value"]:.3f})',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()

    if experiment_path:
        os.makedirs(experiment_path / "imgs", exist_ok=True)
        plt.savefig(experiment_path / "imgs" / f"entropy_acc_scatter_{entropy_percentile}th_percentile_{aggregation_method}.png", dpi=150)

    return plt.gcf()


def parse_experiment(
    exp_name: str,
    model_name: str,
    data_name: str,
    metric="mean",
    to_load="default",          # "entropy"
    percentiles: list = [0, 50, 70, 80, 90, 95, 99, 99.5, 99.9],        # 0 correspond to the entire distribution, 50 to the median value, and then peaks
    plot: bool = True,
    verbose: bool = True,
):
    experiment_path = Path(f"outputs/{model_name}/{data_name}/{exp_name}")

    if metric == "max":
        percentiles = [100]  # max entropy is always the highest peak, so we only need to analyze it once

    def _parse_data(data):
        # inital parsing of the generations
        for i in tqdm(range(len(data)), desc="Parsing generations"):
            data[i]['extracted_answers'] = parse_string_to_list(data[i]['extracted_answers'])

            parsed_entropies = [parse_string_to_list(item) for item in data[i]['entropies']]
            data[i]['entropies'] = parsed_entropies
        return data

    if to_load == "default":
        original_file = experiment_path / f"{exp_name}_default_results.json"
        cached_file = experiment_path / f"{exp_name}_default_results.large"
        
        # Check if cached parsed file exists
        if cached_file.exists():
            print(f"Loading cached parsed data from {cached_file}")
            with open(cached_file, "r") as f:
                data = json.load(f)
            if verbose:
                print(f"Loaded {len(data)} default generations from cached file")
        else:
            # Load original file and parse it
            with open(original_file, "r") as f:
                default_gens = json.load(f)

            data = _parse_data(default_gens["default-generations"])
            
            # Save parsed data to cache file
            print(f"Saving parsed data to cache file: {cached_file}")
            with open(cached_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if verbose:
                print(f"Loaded {len(data)} default generations from {original_file}")

    elif to_load == "entropy" or to_load == "aware":
        matches = list(experiment_path.glob(f"{exp_name}_aware_results*.json"))
        if not matches:
            raise FileNotFoundError(f"No results file found for {exp_name} in {experiment_path}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple result files found for {exp_name}: {matches}")
        
        original_file = matches[0]
        # Create cached filename based on the original filename
        cached_file = original_file.with_suffix('.large')
        
        # Check if cached parsed file exists
        if cached_file.exists():
            print(f"Loading cached parsed data from {cached_file}")
            with open(cached_file, "r") as f:
                data = json.load(f)
            if verbose:
                print(f"Loaded {len(data)} entropy-aware generations from cached file")
        else:
            # Load original file and parse it
            with open(original_file, "r") as f:
                aware_gens = json.load(f)

            data = _parse_data(aware_gens["entropy-aware-generations"])
            
            # Save parsed data to cache file
            print(f"Saving parsed data to cache file: {cached_file}")
            with open(cached_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            if verbose:
                print(f"Loaded {len(data)} entropy-aware generations from {original_file}")

        return data, None
    elif to_load == "more":
        matches = list(experiment_path.glob(f"{exp_name}_aware_more_budget_results*.json"))
        if not matches:
            print(f"No results file found for {exp_name} in {experiment_path}, considering the first")
            return None, None
        elif len(matches) > 1:
            print(f"Multiple result files found for {exp_name}: {matches}")
            return None, None
        
        original_file = matches[0]
        # Create cached filename based on the original filename
        cached_file = original_file.with_suffix('.large')

        # Check if cached parsed file exists
        if cached_file.exists():
            print(f"Loading cached parsed data from {cached_file}")
            with open(cached_file, "r") as f:
                data = json.load(f)
            if verbose:
                print(f"Loaded {len(data)} more-aware generations from cached file")
        else:
            # Load original file and parse it
            with open(original_file, "r") as f:
                aware_gens = json.load(f)

            data = _parse_data(aware_gens["entropy-aware-generations"])

            # Save parsed data to cache file
            print(f"Saving parsed data to cache file: {cached_file}")
            with open(cached_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if verbose:
                print(f"Loaded {len(data)} more-aware generations from {original_file}")

        return data, None
    elif to_load == "more_easy":
        matches = list(experiment_path.glob(f"{exp_name}_aware_easy_more_budget_results*.json"))
        if not matches:
            print(f"No results file found for {exp_name} in {experiment_path}, considering the first")
            return None, None
        elif len(matches) > 1:
            print(f"Multiple result files found for {exp_name}: {matches}")
            return None, None
        original_file = matches[0]
        # Create cached filename based on the original filename
        cached_file = original_file.with_suffix('.large')
        # Check if cached parsed file exists
        if cached_file.exists():
            print(f"Loading cached parsed data from {cached_file}")
            with open(cached_file, "r") as f:
                data = json.load(f)
            if verbose:
                print(f"Loaded {len(data)} easy-more-aware generations from cached file")
        else:
            # Load original file and parse it
            with open(original_file, "r") as f:
                aware_gens = json.load(f)

            data = _parse_data(aware_gens["entropy-aware-generations"])

            # Save parsed data to cache file
            print(f"Saving parsed data to cache file: {cached_file}")
            with open(cached_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if verbose:
                print(f"Loaded {len(data)} easy-more-aware generations from {original_file}")

        return data, None
    else:
        raise ValueError(f"Unknown to_load value: {to_load}. Use 'default', 'entropy' or 'more' or 'easy_more'.")


    if verbose:
        print("Analyzing entropy-accuracy correlation...")
        print(f"Number of samples: {len(data)}")

    all_results = {}
    full_output = f"{experiment_path}, {exp_name}, {model_name}, {data_name}\n\n{metric}\n"
    
    # Analyze different percentiles
    for percentile in tqdm(percentiles, desc="Analyzing percentiles"):
        results = analyze_entropy_accuracy_correlation(
            data, percentile, metric_to_use=METRIC_MAPPING[metric]
        )

        # Create plot
        if plot:
            plt.figure()
            plot_entropy_accuracy_scatter(results, percentile, experiment_path=experiment_path, aggregation_method=metric)
            plt.show()
        
        full_output += REPORT_TEMPLATE.format(
            percentile=percentile,
            n_samples=len(data),

            pearsonr=f"{results['pearson_correlation']:.4f}",
            pearson_p=f"{results['pearson_p_value']:.4f}",
            spearmanr=f"{results['spearman_correlation']:.4f}",
            spearman_p=f"{results['spearman_p_value']:.4f}",

            mean_entropy=np.mean(results['entropy_measures']),
            median_entropy=np.median(results['entropy_measures']),
            max_entropy=np.max(results['entropy_measures']),
            min_entropy=np.min(results['entropy_measures']),

            mean_accuracy=np.mean(results['accuracies']),
            median_accuracy=np.median(results['accuracies']),
            min_accuracy=np.min(results['accuracies']),
            max_accuracy=np.max(results['accuracies']),
        )

        # Store results for this percentile
        all_results[percentile] = results


    if verbose:
        print("\n--- Full Analysis Report ---")
        print(full_output)


    # Save full report to file
    with open(experiment_path / f"entropy_acc_report_{exp_name}_{metric}.txt", "w") as report_file:
        report_file.write(full_output)

    # parsed data and results
    return data, all_results







REPORT_TEMPLATE = """

--- Analysis using {percentile}th percentile entropy peaks ---
Number of valid samples: {n_samples}
Pearson correlation: {pearsonr} (p={pearson_p})
Spearman correlation: {spearmanr} (p={spearman_p})
Mean entropy measure: {mean_entropy:.4f}
Median entropy measure: {median_entropy:.4f}
Max entropy measure: {max_entropy:.4f}
Min entropy (percentile threshold): {min_entropy:.4f}
Mean accuracy: {mean_accuracy:.4f}
Median accuracy: {median_accuracy:.4f}
Entropy range: [{min_entropy:.4f}, {max_entropy:.4f}]
Accuracy range: [{min_accuracy:.4f}, {max_accuracy:.4f}]
"""

METRIC_MAPPING = {      # metric that can be used to aggregate entropy peaks
    "mean": np.mean,
    "median": np.median,
    "max": np.max,      # corresponds to the max entropy recoded (highest peak), percentile here is technically useless
    "min": np.min,      # corresponds to the percentile threshold (exact percentile)
    "std": np.std,
}

# Example usage and testing
def main():
    # working
    # exp_name = "vllm_2025-06-27_05-56-08"
    # experiment_path = f"outputs/DeepSeek-R1-Distill-Qwen-7B/AIME_2024/{exp_name}"

    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    # model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # model_name = "HuggingFaceTB/SmolLM3-3B"
    # model_name = "Qwen/Qwen3-4B"

    # working
    exp_name = "vllm_2025-06-30_21-02-10"
    data_name = "AIME_2024"
    model_name = "DeepSeek-R1-Distill-Qwen-7B"

    # working w/ 90th percentile, statistical not significat with higher percentiles
    exp_name = "2025-07-05_10-33-50"
    data_name = "AIME_2024"
    model_name = "DeepSeek-R1-Distill-Qwen-14B"
    metric = "mean"


    data, all_results = parse_experiment(
        exp_name=exp_name,
        model_name=model_name,
        data_name=data_name,
        metric=metric,
    )


if __name__ == "__main__":
    main()


