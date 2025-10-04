#!/usr/bin/env python3
"""
Script to evaluate code generations from entropy-aware experiments using EvalPlus.
"""

import json
import os
import glob
import tempfile
import subprocess
import sys
import re

from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any

from evalplus.evaluate import evaluate
from evalplus.sanitize import script as sanitize

def find_json_files(model_name: str, data_name: str, id_exp: str) -> dict[str, str]:
    """Find the JSON file matching the pattern."""
    # Convert data_name to match directory structure (replace / with --)

    base_path = f"outputs/{model_name}/{data_name}/{id_exp}"

    default_pattern = f"{base_path}/{id_exp}_default_results.json"
    aware_pattern = f"{base_path}/{id_exp}_aware_results__th*.json"
    more_pattern = f"{base_path}/{id_exp}_aware_more_budget_results__th*.json"
    more_easy_pattern = f"{base_path}/{id_exp}_aware_more_budget_easy_results__th*.json"
    
    default_files = glob.glob(default_pattern)
    aware_files = glob.glob(aware_pattern)
    more_files = glob.glob(more_pattern)
    more_easy_files = glob.glob(more_easy_pattern)
    
    if not default_files and not aware_files and not more_files:
        raise FileNotFoundError(f"No JSON file found matching pattern:")
    
    all_files = {}
    for f in default_files:
        all_files[f] = 'default'
    for f in aware_files:
        all_files[f] = 'aware'
    for f in more_files:
        all_files[f] = 'more'
    for f in more_easy_files:
        all_files[f] = 'more_easy'

    return all_files

def clean_generation_from_think(
        generation: str, 
        end_think_patter: str = "</think>",     # or "final<|message|>"
    ) -> str:
    """Extract clean Python code from a generation string."""
    # Remove any special tokens or prefixes
    result = generation.strip()


    # find </think> and remove everything before it
    think_index = result.find(end_think_patter)
    if think_index != -1:
        result = result[think_index + len(end_think_patter):].strip()
    else:
        result = ""     # if there is a think, we discard the generation


    # check list
    def has_string_over_20(lst):
        counts = {}
        for s in lst:
            counts[s] = counts.get(s, 0) + 1
            if counts[s] > 60:
                return True
        return False
    lines = result.split('\n')
    if has_string_over_20(lines):
        result = "-"     # if there is a think, we discard the generation
    
    return result

def create_samples_file(data: Dict[str, Any], output_file: Path, model_name: str) -> None:
    """Create a samples.jsonl file for EvalPlus evaluation."""
    samples = []
    
    for i, item in tqdm(enumerate(data), desc="Creating samples"):
        task_id = f"HumanEval/{i}"
        
        # Process each generation for this prompt
        for _, generation in enumerate(item["generations"]):
            solution = clean_generation_from_think(
                generation,
                end_think_patter="</think>" if 'gpt' not in model_name else "final<|message|>"
            )
            
            # Create a sample entry
            sample = {
                "task_id": task_id,
                "solution": solution
            }
            samples.append(sample)
    
    # Write samples to JSONL file
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created samples file with {len(samples)} entries: {output_file}")

def run_evalplus_sanitize(samples_file: Path) -> Path:
    """Run EvalPlus sanitization on the samples file."""
    try:
        # result = subprocess.run(
        #     ["evalplus.sanitize", "--samples", samples_file],
        #     capture_output=True,
        #     text=True,
        #     check=True
        # )
        print("#######################################################")
        print("#######################################################")
        print("\t running: evalplus.sanitize --samples", samples_file)
        result = sanitize(samples=str(samples_file))
        print("Sanitization completed successfully!")
        print("#######################################################")
        print("#######################################################")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during sanitization: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

def run_evalplus_evaluate(samples_file: Path) -> None:
    """Run EvalPlus evaluation on the samples file."""
    try:
        print("\t running: evalplus.evaluate --dataset humaneval --samples", samples_file, "--parallel")
        # result = subprocess.run(
        #     ["evalplus.evaluate", "--dataset", "humaneval", "--samples", samples_file, "--parallel"],
        #     capture_output=True,
        #     text=True,
        #     check=True
        # )

        print("#######################################################")
        print("#######################################################")
        print("\t running: evalplus.evaluate --dataset humaneval --samples", samples_file, "--parallel")
        result = evaluate(
            dataset="humaneval",
            samples=str(samples_file),
            parallel=6,
        )
        print("Evaluation completed successfully!")
        print("#######################################################")
        print("#######################################################")
        
        print("Evaluation completed successfully!")
        
        # Parse results from stdout - this is a simplified approach
        # EvalPlus typically outputs pass@1 rates
        return
        
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

def update_json_with_results(
    generation_to_update: Dict[str, Any],
    cache_json: Dict[str, Any],
) -> Dict[str, Any]:
    """Update the original JSON with evaluation results."""

    results = cache_json.get("eval", {})

    # Map task_id to pass status
    extracted_answers = {}
    for task_id, entries in results.items():
        # convert task_id to index: HumanEval/0 -> 0
        idx = int(task_id.split("/")[-1])
        
        extracted_answers = []
        for entry in entries:
            if entry["plus_status"] == "pass":
                extracted_answers.append('1')
            else:
                extracted_answers.append('0')

        generation_to_update[idx]["extracted_answers"] = str(extracted_answers)


    return generation_to_update

def main():
    """Main execution function."""
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_name> <data_name> <id_exp>")
        print("Example: python script.py deepseek-ai--DeepSeek-R1-0528-Qwen3-8B evalplus/humanevalplus 2025-09-12_04-12-36")
        sys.exit(1)
    
    model_name = sys.argv[1]
    data_name = sys.argv[2] 
    id_exp = sys.argv[3]
    
    # Find and load the JSON file
    json_files = find_json_files(model_name, data_name, id_exp)
    print(f"Found JSON files: {json_files}")
    
    for f, ftype in json_files.items():

        with open(f, 'r') as file:
            json_data = json.load(file)
    
        generation_key = 'entropy-aware-generations' if ftype != 'default' else 'default-generations'

        print(f" -> Loaded JSON with {len(json_data[generation_key])} prompts")

        # make a copy of the file (new_name: "backup_{ftype}.json" it will be overwritten later)
        new_file_name = f'backup_{ftype}.json'
        path_backup = Path(f).parent / new_file_name
        with open(path_backup, 'w') as backup_file:
            json.dump(json_data, backup_file, indent=4, ensure_ascii=False)
        print(f"[Backup] of original JSON saved to: {path_backup}")

        base_path = Path(f).parent
        file_name = "_tmp_" + Path(f).name
        samples_file = base_path / file_name.replace(".json", "_samples.jsonl")

        if os.path.exists(samples_file):
            print(f"Samples file already exists: {samples_file}")
        else:
            # Create samples file for EvalPlus
            create_samples_file(json_data[generation_key], samples_file, model_name=model_name)

        # Run sanitization
        print("Running EvalPlus sanitization... \t\t(ETA: ~ 1.5m / 1000 examples)")
        sanitized_file = samples_file.with_name(samples_file.stem + "-sanitized.jsonl")
        if os.path.exists(sanitized_file):
            print(f"Sanitized file already exists: {sanitized_file}")
        else:
            run_evalplus_sanitize(samples_file)
            print(f"Sanitization completed. Output: {sanitized_file}")

        # Run evaluation
        print("Running EvalPlus evaluation... \t\t(ETA: ~ ??.?m / 1000 examples))")
        run_evalplus_evaluate(sanitized_file)
    
        cache_file = str(sanitized_file).replace(".jsonl", "_eval_results.json")

        if not os.path.exists(cache_file):
            print(f"Cache file not found: {cache_file}")
            exit()

        with open(cache_file, 'r') as file:
            cache_json = json.load(file)

        print(cache_json.keys())

        # Update JSON with results
        print("Updating JSON with evaluation results...")
        #             "extracted_answers": "['']",

        updated_gens = update_json_with_results(json_data[generation_key], cache_json)
        # update
        json_data[generation_key] = updated_gens

        # save updated JSON
        output_file = f     # replace file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    
        print(f"Updated JSON saved to: {output_file}")
    
    print("Evaluation completed successfully!")
        

if __name__ == "__main__":
    main()
