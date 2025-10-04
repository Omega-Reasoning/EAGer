from pathlib import Path
import torch
import fire
import os
import random
import time
import json
import gc
from typing import List, Tuple, Dict, Any, Optional
import shutil
import ast

from time import gmtime, strftime
from tqdm import tqdm
from datasets import load_dataset

from src.entropy_generator_vllm import VLLMGenerator as Generator
from src.evaluate import compute_average_accuracy, compute_pass_at_1, compute_cons_at_max, extract_boxed_answer

# set precision to increase performance
torch.set_float32_matmul_precision('high')  # 'medium' or 'high' for better performance

def mock_data() -> Tuple[List[str], List[int]]:
    """Mock data for testing purposes."""
    prompts = [
        "Reply in \\boxed{}. What is the value of 2 + 3?",
        "Reply in \\boxed{}. Calculate 7 - 215",
        "Reply in \\boxed{}. Find the product of 4 and 6.",
        "Reply in \\boxed{}. How tall is the Eiffel Tower?",
    ]
    targets = [5, -208, 24, 330]  # Corresponding answers
    return prompts, targets


def get_data(data_name: str) -> Tuple[List[str], List[int | float | str]]:
    """Load dataset based on data_name."""
    if data_name == "mock": return mock_data()

    if "AIME_2024".lower() in data_name.lower():
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split='train')
        prompts = [text for text in dataset['Problem']]
        targets = [tgt for tgt in dataset['Answer']]
        return prompts, targets

    if "gsm8k".lower() in data_name.lower():
        dataset = load_dataset("openai/gsm8k", "main", split="test")        # 1.3k samples
        prompts_to_select = 1
        prompts = [text for text in dataset['question'][:prompts_to_select]]
        targets = [tgt for tgt in dataset['answer'][:prompts_to_select]]
        # extract exact answer
        targets = [int(answer.split(' ')[-1]) for answer in targets]
        return prompts, targets

    if "AIME2025".lower() in data_name.lower():
        split1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")   # 15 samples
        split2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")  # 15 samples
        split1 = split1.to_dict()
        split2 = split2.to_dict()
        full_questions = split1['question'] + split2['question']
        full_answers = split1['answer'] + split2['answer']
        return full_questions, full_answers

    if "GSM-Plus".lower() in data_name.lower():
        dataset = load_dataset("qintongli/GSM-Plus", split="testmini")  # 2.4k samples
        instruction_plus = "Answer the following question in \\boxed{} format. If you think there is no solution, answer with \\boxed{None}."
        prompts = [instruction_plus + text for text in dataset['question']]
        targets = [tgt for tgt in dataset['answer']]
        return prompts, targets

    if "GPQA".lower() in data_name.lower():
        dataset = load_dataset("fingertap/GPQA-Diamond", split="test")  # 198 samples
        instruction_plus = "Answer the following question using \\boxed{} environment to indicate the letter of the correct solution (example: \\boxed{A})."
        prompts = [instruction_plus + text for text in dataset['question']]
        targets: list[str] = [tgt for tgt in dataset['answer']]
        return prompts, targets

    if "MATH-500".lower() in data_name.lower():
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompts = [text for text in dataset['problem']]
        targets = [tgt for tgt in dataset['answer']]
        return prompts, targets

    
    if "hmmt".lower() in data_name.lower():         #    "MathArena/hmmt_feb_2025"
        dataset = load_dataset("MathArena/hmmt_feb_2025", split="train")  # 30 samples
        prompts = [text for text in dataset['problem']]
        targets = [tgt for tgt in dataset['answer']]
        return prompts, targets

    if "humanevalplus".lower() in data_name.lower():      # evalplus/humanevalplus
        dataset = load_dataset("evalplus/humanevalplus", split="test")
        prompts = [text for text in dataset['prompt']]
        targets = ['1' for _ in dataset['prompt']]      # fake label, need to verify with external evaluator
        return prompts, targets

    raise ValueError(f"Unknown dataset: {data_name}. Available: ['mock', 'Maxwell-Jia/AIME_2024']")


def save_incremental_results(
    output_file: Path,
    parameters_log: Dict[str, Any],
    generations: List[List[str]],
    entropies: List[List[List[float]]],
    prompts: List[str],
    targets: List[int],
    notes: str,
    generation_time: float,
    recorded_branches: Optional[List[List[Dict]]] = None,
    status: str = "IN_PROGRESS",
    completed_sequences: int = 0,
    total_sequences: int = 0
) -> None:
    """Save results incrementally to JSON file."""
    print(f"Saving results to {output_file} ...")
    # Round entropies for readability
    entropies_approx = [
        [[round(entropy, 4) for entropy in seq_entropy] for seq_entropy in prompt_entropies] 
        for prompt_entropies in entropies
    ]
    entropies_approx_str = [
        [str(seq_entropy) for seq_entropy in prompt_entropies] 
        for prompt_entropies in entropies_approx
    ]
    
    # Compute current metrics
    avg_acc = compute_average_accuracy(generations, targets)
    pass_at_1 = compute_pass_at_1(generations, targets)
    cons_at_max = compute_cons_at_max(generations, targets)
    
    output_log = {
        **parameters_log,
        "status": {
            "state": status,
            "completed_sequences": completed_sequences,
            "total_sequences": total_sequences,
            "progress": f"{completed_sequences}/{total_sequences}" if total_sequences > 0 else "0/0"
        },
        "extra": {
            "notes": notes,
            "generation_time_so_far (s)": generation_time,
        },
        "results": {
            "avg_acc": avg_acc,
            "pass_at_1": pass_at_1,
            "cons_at_max": cons_at_max,
        },
    }

    # Save only the output_log to keep track of the experiment status, without the acutal generations to be saved later
    # this file ends with .jsonl, because .json files are in gitignore
    with open(output_file.with_suffix('.jsonl'), "w") as f:
        json.dump(output_log, f, indent=4, ensure_ascii=False)
    
    # Add generation data based on whether it's default or entropy-aware
    if recorded_branches is None:
        # Default generation
        output_log["default-generations"] = [
            {
                "prompt": prompts[i],
                "generated_sequences": len(generations[i]),
                "target": targets[i],
                "extracted_answers": str([extract_boxed_answer(gen) for gen in generations[i]]),
                "generations": generations[i],
                "entropies": entropies_approx_str[i],
            }
            for i in range(len(generations))
        ]
    else:
        # Entropy-aware generation
        output_log["entropy-aware-generations"] = [
            {
                "prompt": prompts[i],
                "generated_sequences": len(generations[i]),
                "target": targets[i],
                "extracted_answers": str([extract_boxed_answer(gen) for gen in generations[i]]),
                "generations": generations[i],
                "entropies": entropies_approx_str[i],
                "recorded_branches": [str(dict_info) for dict_info in recorded_branches[i]],        # branch history here
            }
            for i in range(len(generations))
        ]
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(output_log, f, indent=4, ensure_ascii=False)
    except Exception as e:
        # save with torch.save
        torch.save(output_log, output_file.with_suffix('.dict'))

        print(f'####################################')
        print(f'####################################')
        print(f"Error saving results to {output_file}: {e}")
        print(f'####################################')
        print(f'####################################')
        print(f'Saving as .dict file instead, convert it to json manually.')
        print(f'####################################')
        print(f'####################################')


def run_and_evaluate_default(
    generator: Generator,
    prompts: List[str],
    targets: List[int],
    output_file: Path,
    parameters_log: Dict[str, Any],
    notes: str,
    temperature: float = 0.7,
    max_sequences: int = 16,
    max_model_len: int = 32_768,
    existing_generations: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[List[str]], List[List[List[float]]], float, float, float, float]:
    """Run the default evaluation without entropy-aware generation."""
    start_time = time.time()
    default_generations = []
    default_entropies = []
    
    total_prompts = len(prompts)
    
    for i, prompt in tqdm(enumerate(prompts), desc="Generating default", dynamic_ncols=True, total=total_prompts):
        if existing_generations is not None and i < len(existing_generations):
            # If we have existing generations, skip this prompt
            print(f"Skipping prompt {i}/{total_prompts} as it already has existing generations.")
            default_generations.append(existing_generations[i]["generations"])
            default_entropies.append(
                [ast.literal_eval(x) for x in existing_generations[i]["entropies"]]     # are saved as strings
            )

        else: 
            print(f"Generating prompt {i}/{total_prompts}:")
            generations, entropies = generator.default_generation(
                prompt=prompt,
                temperature=temperature,
                max_sequences=max_sequences,
                max_new_tokens=max_model_len,
            )
            
            default_generations.append(generations)
            default_entropies.append(entropies)
        
        # Save incrementally every generation
        current_time = time.time() - start_time
        status = "COMPLETED" if i == total_prompts - 1 else "IN_PROGRESS"
        
        save_incremental_results(
            output_file=output_file,
            parameters_log=parameters_log,
            generations=default_generations,
            entropies=default_entropies,
            prompts=prompts,
            targets=targets,
            notes=notes,
            generation_time=current_time,
            status=status,
            completed_sequences=i + 1,
            total_sequences=total_prompts,
        )
    
    end_time = time.time() - start_time
    
    # Compute final metrics
    avg_acc = compute_average_accuracy(default_generations, targets)
    pass_at_1 = compute_pass_at_1(default_generations, targets)
    cons_at_max = compute_cons_at_max(default_generations, targets)
    
    print(f"Pass@1: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
    print(f"Cons@{max_sequences}: {cons_at_max:.4f} ({cons_at_max*100:.2f}%)")
    
    return default_generations, default_entropies, avg_acc, pass_at_1, cons_at_max, end_time


def run_and_evaluate_entropy(
    generator: Generator,
    prompts: List[str],
    targets: List[int],
    output_file: Path,
    parameters_log: Dict[str, Any],
    notes: str,
    temperature: float = 0.7,
    entropy_threshold: float = 2.5,
    max_sequences: int = 16,
    max_model_len: int = 32_768,
    existing_generations: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[List[str]], List[List[List[float]]], List[List[Dict]], float, float, float, float]:
    """Run entropy-aware evaluation with incremental saving."""
    start_time = time.time()
    aware_generations: List[List[str]] = []
    aware_entropies: List[List[List[float]]] = []
    aware_recorded_branches: List[List[Dict]] = []
    
    total_prompts = len(prompts)
    
    for i, prompt in tqdm(enumerate(prompts), desc="Generating entropy aware", dynamic_ncols=True, total=total_prompts):
        if existing_generations is not None and i < len(existing_generations):
            # If we have existing generations, skip this prompt
            print(f"Skipping prompt {i}/{total_prompts} as it already has existing generations.")
            aware_generations.append(existing_generations[i]["generations"])
            aware_entropies.append(
                [ast.literal_eval(x) for x in existing_generations[i]["entropies"]]     # are saved as strings
            )
            aware_recorded_branches.append(
                [ast.literal_eval(x) for x in existing_generations[i]["recorded_branches"]]  # are saved as strings
            )
        else:
            print(f"Generating prompt {i}/{total_prompts}:")
            generations, entropies, recorded_branches = generator.entropy_aware_generation(
                prompt=prompt,
                temperature=temperature,
                entropy_threshold=entropy_threshold,
                initial_sequences=1,
                max_sequences=max_sequences,
                max_new_tokens=max_model_len,
                max_tokens_without_branch=1000,
                verbose=True,
            )
        
            aware_generations.append(generations)
            aware_entropies.append(entropies)
            aware_recorded_branches.append(recorded_branches)
        
        # Save incrementally every generation
        current_time = time.time() - start_time
        status = "COMPLETED" if i == total_prompts - 1 else "IN_PROGRESS"
        
        save_incremental_results(
            output_file=output_file,
            parameters_log=parameters_log,
            generations=aware_generations,
            entropies=aware_entropies,
            prompts=prompts,
            targets=targets,
            notes=notes,
            generation_time=current_time,
            recorded_branches=aware_recorded_branches,
            status=status,
            completed_sequences=i + 1,
            total_sequences=total_prompts
        )
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
    
    end_time = time.time() - start_time
    
    # Compute final metrics
    avg_acc = compute_average_accuracy(aware_generations, targets)
    pass_at_1 = compute_pass_at_1(aware_generations, targets)
    cons_at_max = compute_cons_at_max(aware_generations, targets)
    
    print(f"Pass@1: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
    print(f"Cons@{max_sequences}: {cons_at_max:.4f} ({cons_at_max*100:.2f}%)")

    return aware_generations, aware_entropies, aware_recorded_branches, avg_acc, pass_at_1, cons_at_max, end_time

def run_and_evaluate_entropy_with_budget_easy(
    generator: Generator,
    prompts: List[str],
    targets: List[int],
    input_file: Path,
    output_file_more_budget: Path,
    notes: str,
    temperature: float = 0.6,
    entropy_threshold: float = 2.5,
    max_model_len: int = 32_768,
    start_from: int | None = None,
):
    # read input file
    with open(input_file, 'r') as f:
        existing_results = json.load(f)

    generations_info: List[Dict[str, Any]] = existing_results.get("entropy-aware-generations", [])
    aware_generations: List[List[str]] = [
        gen["generations"] for gen in generations_info
    ]
    aware_entropies: List[List[List[float]]] = [
        [ast.literal_eval(x) for x in gen["entropies"]]
        for gen in generations_info
    ]
    aware_recorded_branches: List[List[Dict]] = [
        [ast.literal_eval(x) for x in gen["recorded_branches"]]
        for gen in generations_info
    ]

    aware_extracted_answers: List[List[str]] = [
        ast.literal_eval(gen["extracted_answers"])
        for gen in generations_info
    ]

    # Run prompt with additional budget
    aware_generations_budget: List[List[str]] = []
    aware_entropies_budget: List[List[List[float]]] = []
    aware_recorded_branches_budget: List[List[Dict]] = []

    parameters_log = existing_results.get("params", {})
    total_prompts = len(prompts)
    max_sequences = existing_results["params"]["max_sequences"]

    # compute budget
    num_generated_sequences = sum(len(gens) for gens in aware_generations)
    total_theoretical_budget = max_sequences * total_prompts
    budget_remaining = total_theoretical_budget - num_generated_sequences
    assert total_theoretical_budget >= num_generated_sequences, \
        f"Total theoretical budget {total_theoretical_budget} is less than generated sequences {num_generated_sequences}. Something's wrong."
    
    idx_to_augment_bit_more = []        # sequences with less than max_sequences, give them another try
    idx_to_augment_plus_max = []        # sequences with max_sequences, but still not reaching the target answer
    for i, (current_gens, target) in enumerate(zip(aware_generations, targets)):
        # extracted_answers = [extract_boxed_answer(g) for g in current_gens]
        extracted_answers = aware_extracted_answers[i]

        # if str(target) not in extracted_answers and str(target).lower() != "none":
        #     # none of the current generations contain the target answer
        #     # pass@1 = 0
        #     if len(current_gens) < max_sequences:
        #         idx_to_augment_bit_more.append(i)
        #     else:
        #         idx_to_augment_plus_max.append(i)

        if len(extracted_answers) == 32:
            idx_to_augment_plus_max.append(i)


    print(f"Prompts needing additional budget:")
    print(f"\t- {len(idx_to_augment_plus_max)}/{total_prompts} prompts with max_sequences ({max_sequences}) but no pass@1 == 1.")
    print(f"\t- {len(idx_to_augment_bit_more)}/{total_prompts} prompts with less than max_sequences ({max_sequences}) and no pass@1 == 1.")
    print(f"\t- Budget remaining: {budget_remaining} sequences.")

    idx_with_more_budget = idx_to_augment_plus_max + idx_to_augment_bit_more
    
    distribution_table = None
    if budget_remaining == 0:
        print(f"No budget remaining, all prompts have pass@1 == 1 or max_sequences reached.")

    elif len(idx_with_more_budget) == 0:
        print(f"No prompts need additional budget, all prompt sequences have pass@1 == 1.")

    elif budget_remaining >= len(idx_with_more_budget):
        print(f"Budget remaining ({budget_remaining}) is enough to cover all prompts needing more budget ({len(idx_with_more_budget)}).")

        # distribute budget evenly
        budget_per_prompt = budget_remaining // len(idx_with_more_budget)
        distribution_table = {idx: b for idx, b in zip(idx_with_more_budget, [budget_per_prompt] * len(idx_with_more_budget))}

        rest_budget = budget_remaining % len(idx_with_more_budget)
        if rest_budget > 0:
            for idx in idx_with_more_budget[:rest_budget]:
                distribution_table[idx] += 1

    elif budget_remaining < len(idx_with_more_budget):
        print(f"Budget remaining ({budget_remaining}) is less than the number of prompts needing more budget ({len(idx_with_more_budget)}).")
        print(f"Distributing budget only on the first {budget_remaining} prompts needing more budget.")
        # distribute budget evenly, but only on the first `budget_remaining` prompts
        distribution_table = {idx: 1 for idx in idx_with_more_budget[:budget_remaining]}

    print(f"Distribution table: {distribution_table}")

    start_time = time.time()
    
    if distribution_table:

        if start_from:
            sequences_to_skip = [i for i in range(start_from)]
        else:
            sequences_to_skip = []

        for i, (generations, entropies, branches) in enumerate(zip(aware_generations, aware_entropies, aware_recorded_branches)):
            if i in sequences_to_skip:
                print(f"Skipping prompt {i}/{total_prompts} as per start_from={start_from}.")
                aware_generations_budget.append(generations)
                aware_entropies_budget.append(entropies)
                aware_recorded_branches_budget.append(branches)
                continue

            print(f"\nProcessing prompt {i}/{total_prompts}:")

            aware_generations_budget.append(generations)
            aware_entropies_budget.append(entropies)
            aware_recorded_branches_budget.append(branches)

            # reset entropy threshold
            new_entropy_threshold = entropy_threshold

            if i in distribution_table: # needs additional budget
                additional_sequences = min(distribution_table[i], int(3*max_sequences))
                print(f"Prompt {i} needs additional {additional_sequences} sequences.")

                new_generations, new_entropies, new_recorded_branches = [], [], []

                new_entropy_threshold = entropy_threshold
                
                tries = 0
                while len(new_generations) < additional_sequences:
                    tries += 1
                    # lower entropy if the prompt is in idx_to_augment_bit_more (i.e. previous threshold didn't allow for enough generations/exploration)
                    # else, keep the same entropy threshold
                    print(f"Generating additional sequences for prompt {i} ({len(new_generations)}/{additional_sequences})")
                    new_entropy_threshold = new_entropy_threshold - (new_entropy_threshold * 0.2) if i in idx_to_augment_bit_more else entropy_threshold
                    if tries > 2:
                        new_entropy_threshold = new_entropy_threshold - (new_entropy_threshold * 0.1)
                    print(f"Using entropy threshold: {new_entropy_threshold:.2f} for prompt {i}; (original: {entropy_threshold:.2f})")

                    turn_generations, turn_entropies, turn_branches = generator.entropy_aware_generation(
                        prompt=prompts[i],
                        temperature=temperature,
                        entropy_threshold=new_entropy_threshold,
                        initial_sequences=1,
                        max_sequences=additional_sequences,
                        max_new_tokens=max_model_len,
                        max_tokens_without_branch=1000,
                        verbose=True,
                    )
                    new_generations.extend(turn_generations)
                    new_entropies.extend(turn_entropies)
                    new_recorded_branches.extend(turn_branches)

                # Append additional generations, entropies and branches
                aware_generations_budget[i].extend(new_generations)
                aware_entropies_budget[i].extend(new_entropies)
                aware_recorded_branches_budget[i].extend(new_recorded_branches)
                
                # Clean up memory
                gc.collect()
                torch.cuda.empty_cache()
    
            save_incremental_results(
                output_file=output_file_more_budget,
                parameters_log=parameters_log,
                generations=aware_generations_budget,
                entropies=aware_entropies_budget,
                prompts=prompts,
                targets=targets,
                notes=notes + "-- (with additional budget | {prompt_idx: +seq_num} = " + str(distribution_table) + ")",
                generation_time=time.time() - start_time,
                recorded_branches=aware_recorded_branches_budget,
                status="IN_PROGRESS",
                completed_sequences=i + 1,
                total_sequences=total_prompts
            )

    end_time = time.time() - start_time
        
    # Final metrics for additional budget
    avg_acc_budget = compute_average_accuracy(aware_generations_budget, targets)
    pass_at_1_budget = compute_pass_at_1(aware_generations_budget, targets)
    cons_at_max_budget = compute_cons_at_max(aware_generations_budget, targets)
    
    print(f"Final metrics with additional budget:")
    print(f"Average accuracy (with additional budget): {avg_acc_budget:.4f} ({avg_acc_budget*100:.2f}%)")
    print(f"Pass@1 (with additional budget): {pass_at_1_budget:.4f} ({pass_at_1_budget*100:.2f}%)")
    print(f"Cons@{max_sequences} (with additional budget): {cons_at_max_budget:.4f} ({cons_at_max_budget*100:.2f}%)")
    
    return aware_generations_budget, aware_entropies_budget, aware_recorded_branches_budget, avg_acc_budget, pass_at_1_budget, cons_at_max_budget, end_time


def run_and_evaluate_entropy_with_budget(
    generator: Generator,
    prompts: List[str],
    targets: List[int],
    input_file: Path,
    output_file_more_budget: Path,
    notes: str,
    temperature: float = 0.7,
    entropy_threshold: float = 2.5,
    max_model_len: int = 32_768,
    start_from: int | None = None,
):
    # read input file
    with open(input_file, 'r') as f:
        existing_results = json.load(f)

    generations_info: List[Dict[str, Any]] = existing_results.get("entropy-aware-generations", [])
    aware_generations: List[List[str]] = [
        gen["generations"] for gen in generations_info
    ]
    aware_entropies: List[List[List[float]]] = [
        [ast.literal_eval(x) for x in gen["entropies"]]
        for gen in generations_info
    ]
    aware_recorded_branches: List[List[Dict]] = [
        [ast.literal_eval(x) for x in gen["recorded_branches"]]
        for gen in generations_info
    ]

    aware_extracted_answers: List[List[str]] = [
        ast.literal_eval(gen["extracted_answers"])
        for gen in generations_info
    ]

    # Run prompt with additional budget
    aware_generations_budget: List[List[str]] = []
    aware_entropies_budget: List[List[List[float]]] = []
    aware_recorded_branches_budget: List[List[Dict]] = []

    parameters_log = existing_results.get("params", {})
    total_prompts = len(prompts)
    max_sequences = existing_results["params"]["max_sequences"]

    # compute budget
    num_generated_sequences = sum(len(gens) for gens in aware_generations)
    total_theoretical_budget = max_sequences * total_prompts
    budget_remaining = total_theoretical_budget - num_generated_sequences
    assert total_theoretical_budget >= num_generated_sequences, \
        f"Total theoretical budget {total_theoretical_budget} is less than generated sequences {num_generated_sequences}. Something's wrong."
    
    idx_to_augment_bit_more = []        # sequences with less than max_sequences, give them another try
    idx_to_augment_plus_max = []        # sequences with max_sequences, but still not reaching the target answer
    for i, (current_gens, target) in enumerate(zip(aware_generations, targets)):
        # extracted_answers = [extract_boxed_answer(g) for g in current_gens]
        extracted_answers = aware_extracted_answers[i]

        # print(f"{extracted_answers = }")
        # print(f"{target = }")
        # print(f'{target in extracted_answers = }')
        # exit()

        if str(target) not in extracted_answers and str(target).lower() != "none":
            # none of the current generations contain the target answer
            # pass@1 = 0
            if len(current_gens) < max_sequences:
                idx_to_augment_bit_more.append(i)
            else:
                idx_to_augment_plus_max.append(i)

    print(f"Prompts needing additional budget:")
    print(f"\t- {len(idx_to_augment_plus_max)}/{total_prompts} prompts with max_sequences ({max_sequences}) but no pass@1 == 1.")
    print(f"\t- {len(idx_to_augment_bit_more)}/{total_prompts} prompts with less than max_sequences ({max_sequences}) and no pass@1 == 1.")
    print(f"\t- Budget remaining: {budget_remaining} sequences.")

    idx_with_more_budget = idx_to_augment_plus_max + idx_to_augment_bit_more
    
    distribution_table = None
    if budget_remaining == 0:
        print(f"No budget remaining, all prompts have pass@1 == 1 or max_sequences reached.")

    elif len(idx_with_more_budget) == 0:
        print(f"No prompts need additional budget, all prompt sequences have pass@1 == 1.")

    elif budget_remaining >= len(idx_with_more_budget):
        print(f"Budget remaining ({budget_remaining}) is enough to cover all prompts needing more budget ({len(idx_with_more_budget)}).")

        # distribute budget evenly
        budget_per_prompt = budget_remaining // len(idx_with_more_budget)
        distribution_table = {idx: b for idx, b in zip(idx_with_more_budget, [budget_per_prompt] * len(idx_with_more_budget))}

        rest_budget = budget_remaining % len(idx_with_more_budget)
        if rest_budget > 0:
            for idx in idx_with_more_budget[:rest_budget]:
                distribution_table[idx] += 1

    elif budget_remaining < len(idx_with_more_budget):
        print(f"Budget remaining ({budget_remaining}) is less than the number of prompts needing more budget ({len(idx_with_more_budget)}).")
        print(f"Distributing budget only on the first {budget_remaining} prompts needing more budget.")
        # distribute budget evenly, but only on the first `budget_remaining` prompts
        distribution_table = {idx: 1 for idx in idx_with_more_budget[:budget_remaining]}

    print(f"Distribution table: {distribution_table}")

    start_time = time.time()
    
    if distribution_table:

        if start_from:
            sequences_to_skip = [i for i in range(start_from)]
        else:
            sequences_to_skip = []

        for i, (generations, entropies, branches) in enumerate(zip(aware_generations, aware_entropies, aware_recorded_branches)):
            if i in sequences_to_skip:
                print(f"Skipping prompt {i}/{total_prompts} as per start_from={start_from}.")
                aware_generations_budget.append(generations)
                aware_entropies_budget.append(entropies)
                aware_recorded_branches_budget.append(branches)
                continue

            print(f"\nProcessing prompt {i}/{total_prompts}:")

            aware_generations_budget.append(generations)
            aware_entropies_budget.append(entropies)
            aware_recorded_branches_budget.append(branches)

            # reset entropy threshold
            new_entropy_threshold = entropy_threshold

            if i in distribution_table: # needs additional budget
                additional_sequences = min(distribution_table[i], int(1*max_sequences))
                print(f"Prompt {i} needs additional {additional_sequences} sequences.")

                new_generations, new_entropies, new_recorded_branches = [], [], []

                new_entropy_threshold = entropy_threshold
                
                tries = 0
                while len(new_generations) < additional_sequences:
                    tries += 1
                    # lower entropy if the prompt is in idx_to_augment_bit_more (i.e. previous threshold didn't allow for enough generations/exploration)
                    # else, keep the same entropy threshold
                    print(f"Generating additional sequences for prompt {i} ({len(new_generations)}/{additional_sequences})")
                    new_entropy_threshold = new_entropy_threshold - (new_entropy_threshold * 0.2) if i in idx_to_augment_bit_more else entropy_threshold
                    if tries > 2:
                        new_entropy_threshold = new_entropy_threshold - (new_entropy_threshold * 0.1)
                    print(f"Using entropy threshold: {new_entropy_threshold:.2f} for prompt {i}; (original: {entropy_threshold:.2f})")

                    turn_generations, turn_entropies, turn_branches = generator.entropy_aware_generation(
                        prompt=prompts[i],
                        temperature=temperature,
                        entropy_threshold=new_entropy_threshold,
                        initial_sequences=1,
                        max_sequences=additional_sequences,
                        max_new_tokens=max_model_len,
                        max_tokens_without_branch=1000,
                        verbose=True,
                    )
                    new_generations.extend(turn_generations)
                    new_entropies.extend(turn_entropies)
                    new_recorded_branches.extend(turn_branches)

                # Append additional generations, entropies and branches
                aware_generations_budget[i].extend(new_generations)
                aware_entropies_budget[i].extend(new_entropies)
                aware_recorded_branches_budget[i].extend(new_recorded_branches)
                
                # Clean up memory
                gc.collect()
                torch.cuda.empty_cache()
    
            save_incremental_results(
                output_file=output_file_more_budget,
                parameters_log=parameters_log,
                generations=aware_generations_budget,
                entropies=aware_entropies_budget,
                prompts=prompts,
                targets=targets,
                notes=notes + "-- (with additional budget | {prompt_idx: +seq_num} = " + str(distribution_table) + ")",
                generation_time=time.time() - start_time,
                recorded_branches=aware_recorded_branches_budget,
                status="IN_PROGRESS",
                completed_sequences=i + 1,
                total_sequences=total_prompts
            )

    end_time = time.time() - start_time
        
    # Final metrics for additional budget
    avg_acc_budget = compute_average_accuracy(aware_generations_budget, targets)
    pass_at_1_budget = compute_pass_at_1(aware_generations_budget, targets)
    cons_at_max_budget = compute_cons_at_max(aware_generations_budget, targets)
    
    print(f"Final metrics with additional budget:")
    print(f"Average accuracy (with additional budget): {avg_acc_budget:.4f} ({avg_acc_budget*100:.2f}%)")
    print(f"Pass@1 (with additional budget): {pass_at_1_budget:.4f} ({pass_at_1_budget*100:.2f}%)")
    print(f"Cons@{max_sequences} (with additional budget): {cons_at_max_budget:.4f} ({cons_at_max_budget*100:.2f}%)")
    
    return aware_generations_budget, aware_entropies_budget, aware_recorded_branches_budget, avg_acc_budget, pass_at_1_budget, cons_at_max_budget, end_time


def create_parameters_log(
    experiment_name: str,
    model_name: str,
    temperature: float,
    entropy_threshold: float,
    max_sequences: int,
    seed: int,
    data_name: str,
    prompts: List[str],
    targets: List[int]
) -> Dict[str, Any]:
    """Create the parameters log dictionary."""
    return {
        "experiment_name": experiment_name,
        "params": {
            "model_name": model_name,
            "temperature": temperature,
            "entropy_threshold": entropy_threshold,
            "max_sequences": max_sequences,
            "seed": seed,
        },
        "data": {
            "data_name": data_name,
            "num_prompts": len(prompts),
            "targets": targets,
        },
    }


def main(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    data_name: str = "gsm8k",                                       # "Maxwell-Jia/AIME_2024",
    temperature: float = 0.6,
    entropy_threshold: float | List[float] = 2.5,
    max_sequences: int = 32,
    notes: str = "",
    dtype: str = "bfloat16",
    experiments: str = "all",       # "all", "parallel", "eager_init", "eager_adapt", "eager" | (old: "default", "entropy", "more", "more_easy")
    max_model_len: int = 32_768,
    gpu_memory_utilization: float = 0.7,
    gpu_num: int | None = None,     # auto
    device: str = "cuda",
    output_dir: str | None = None,
    seed: int = 55,
    start_from: int | None = None,   # used only for "more" experiment
):
    # remap experiments name to old names
    # parallel -> default
    # eager_init -> entropy
    # eager_adapt - > more_easy
    # eager -> more
    if experiments == "parallel":
        experiments = "default"
    elif experiments == "eager_init":
        experiments = "entropy"
    elif experiments == "eager_adapt":
        experiments = "more_easy"
    elif experiments == "eager":
        experiments = "more"

    """Main function to run experiments."""
    # Set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if isinstance(entropy_threshold, float):
        e_thresholds: List[float] = [entropy_threshold]
    elif isinstance(entropy_threshold, list) or isinstance(entropy_threshold, tuple):
        e_thresholds: List[float] = entropy_threshold
    else:
        raise ValueError(
            f"Invalid type for entropy_threshold: {type(entropy_threshold)}. Expected float or list of floats."
        )
    
    if output_dir is None:
        experiment_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        experiment_dir = Path("outputs") / model_name.split("/")[-1] / data_name.split("/")[-1] / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
    else:
        experiment_dir = Path(output_dir)
        experiment_name = experiment_dir.name

    print(f"Experiment directory: {experiment_dir}")

    # Initialize generator
    generator = Generator(
        model_name=model_name,
        max_model_len=max_model_len,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_num=gpu_num,
        seed=seed,
    )
    
    # Load data
    prompts, targets = get_data(data_name=data_name)
    
    # Create parameters log
    parameters_log = create_parameters_log(
        experiment_name=experiment_name,
        model_name=model_name,
        temperature=temperature,
        entropy_threshold=e_thresholds[0],
        max_sequences=max_sequences,
        seed=seed,
        data_name=data_name,
        prompts=prompts,
        targets=targets
    )
    
    # Run experiments
    if experiments in ["all", "default"]:
        output_file = experiment_dir / f"{experiment_name}_default_results.json"
        print(f"Running default experiment. Results will be saved to: {output_file}")

        # check if file already exists
        if output_file.exists():
            print(f"\tOutput file {output_file} already exists, loading ...")

            # backup_file = output_file.with_name(output_file.stem + "_backup.jsonb")
            # shutil.copy(output_file, backup_file)
            # print(f"\tBackup created: {backup_file}")

            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            existing_generations = existing_results.get("default-generations", [])
            if existing_generations:
                print(f"\tFound {len(existing_generations)} existing generations in {output_file}.")
        else:
            existing_generations = None
        
        run_and_evaluate_default(
            generator=generator,
            prompts=prompts,
            targets=targets,
            output_file=output_file,
            parameters_log=parameters_log,
            notes=notes,
            temperature=temperature,
            max_sequences=max_sequences,
            max_model_len=max_model_len,
            existing_generations=existing_generations,
        )
    
    if experiments in ["all", "entropy"]:
        for e_th in e_thresholds:
            # update log
            parameters_log["params"]["entropy_threshold"] = e_th

            additional_info = f"_th{e_th}"
            output_file = experiment_dir / f"{experiment_name}_aware_results_{additional_info}.json"
            
            # check if file already exists
            if output_file.exists():
                print(f"\tOutput file {output_file} already exists, loading ...")

                backup_file = output_file.with_name(output_file.stem + "_backup.json")
                shutil.copy(output_file, backup_file)
                print(f"\tBackup created: {backup_file}")

                with open(output_file, 'r') as f:
                    existing_results = json.load(f)
                existing_generations = existing_results.get("entropy-aware-generations", [])
                if existing_generations:
                    print(f"\tFound {len(existing_generations)} existing generations in {output_file}.")
            else:
                existing_generations = None
                print(f"Running entropy-aware experiment ({e_th}). Results will be saved to: {output_file}")

            run_and_evaluate_entropy(
                generator=generator,
                prompts=prompts,
                targets=targets,
                output_file=output_file,
                parameters_log=parameters_log,
                notes=notes,
                temperature=temperature,
                entropy_threshold=e_th,
                max_sequences=max_sequences,
                max_model_len=max_model_len,
                existing_generations=existing_generations,
            )

    if experiments in ["all", "more", "more_easy"]:
        # verify that experiment_dir already exists
        if not experiment_dir.exists():
            raise ValueError(f"Experiment directory {experiment_dir} does not exist. Run entropy-aware experiment first.")

        # extract experiment name from experiment_dir
        # there should be only one json file
        json_files = list(experiment_dir.glob(f"{experiment_name}_aware_results*.json"))
        # remove any backup file
        json_files = [f for f in json_files if "backup" not in f.name]
        # remove samples files
        json_files = [f for f in json_files if "samples" not in f.name]
        if len(json_files) != 1:
            raise ValueError(f"Expected exactly one JSON file in {experiment_dir}, found {len(json_files)}.")
        input_file = json_files[0]
        print(f"Running entropy-aware experiment with additional budget. Input file: {input_file}")
        
        # extract threshold
        e_th = input_file.stem.split('_')[-1].replace("th", "")
        additional_info = f"_th{e_th}"
        output_file = experiment_dir / f"{experiment_name}_aware_results_{additional_info}.json"
        if start_from:
            output_file_more_budget = experiment_dir / f"{experiment_name}_aware_ADDON{start_from}_more_budget_results_{additional_info}.json"
        elif experiments == "more_easy":
            output_file_more_budget = experiment_dir / f"{experiment_name}_aware_easy_more_budget_results_{additional_info}.json"
        else:
            output_file_more_budget = experiment_dir / f"{experiment_name}_aware_more_budget_results_{additional_info}.json"

        print(f"Results will be saved to: {output_file_more_budget}")

        if experiments == "more_easy":
            run_and_evaluate_entropy_with_budget_easy(
                generator=generator,
                prompts=prompts,
                targets=targets,
                input_file=input_file,
                output_file_more_budget=output_file_more_budget,
                notes=notes,
                temperature=temperature,
                entropy_threshold=float(e_th),
                max_model_len=max_model_len,
                start_from=start_from,
            )
        else:
            run_and_evaluate_entropy_with_budget(
                generator=generator,
                prompts=prompts,
                targets=targets,
                input_file=input_file,
                output_file_more_budget=output_file_more_budget,
                notes=notes,
                temperature=temperature,
                entropy_threshold=float(e_th),
                max_model_len=max_model_len,
                start_from=start_from,
            )


if __name__ == "__main__":
    fire.Fire(main)
