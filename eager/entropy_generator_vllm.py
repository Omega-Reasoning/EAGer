import torch
import torch.nn.functional as F
from rich import print
from rich.live import Live
from rich.table import Table
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from tqdm import tqdm
import os

from vllm import LLM, SamplingParams, EngineArgs, LLMEngine
from vllm.outputs import RequestOutput
from transformers import AutoTokenizer


class VLLMModelWrapper:
    def __init__(
        self, 
        model_name: str, 
        tensor_parallel_size: int = 1, 
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.7,
        gpu_num: int | None = None,
        device: str = "cuda",
        seed: int = 55,
    ):
        print(f"Loading model with vLLM: {model_name}")
        download_dir = os.environ.get('HF_HOME', None)
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
            enforce_eager=False,  # Use CUDA graphs for better performance
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,          # Should speed up the generation while in branching mode (storing the KV cache of the prev. prompt)
            download_dir=download_dir,
            # device=device,        # Deprecated by vllm?
            # seed=seed,
        )

        self.seed = seed
        
        # Get tokenizer for utility functions
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded successfully")

    def compute_entropy_from_logprobs(self, logprobs_dict: dict) -> float:
        """
        Compute entropy from vLLM's logprobs output.
        Args:
            logprobs_dict: Dictionary of token_id -> logprob from vLLM
        Returns:
            Entropy value
        """
        if not logprobs_dict:
            # in case of empty logprobs
            return -3.0
            
        # Convert logprobs to probs
        try:
            logprob_values = []
            for token_id, logprob_obj in logprobs_dict.items():
                # vLLM returns Logprob objects with .logprob attribute
                if hasattr(logprob_obj, 'logprob'):
                    logprob_values.append(logprob_obj.logprob)
                else:
                    # Fallback: assume it's already a float
                    logprob_values.append(float(logprob_obj))
            
            if not logprob_values:
                return -4.0
                
            # Convert to tensor
            logprobs = torch.tensor(logprob_values, dtype=torch.float32)
            probs = torch.exp(logprobs)
            
            # Normalize probabilities to sum to 1 (rough approximation)
            # This assumes the remaining probability mass is negligible
            probs = probs / probs.sum()
            
            # Compute entropy over normalized top-K probabilities
            epsilon = 1e-10
            entropy = -(probs * torch.log(probs + epsilon)).sum().item()

            return entropy

        except Exception as e:
            print(f"Warning: Error computing entropy from logprobs: {e}")
            return -5.0

    def initial_tokenization(self, prompt: str) -> str:
        """Format prompt using chat template if available."""
        if self.tokenizer.chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            print("Warning: Using default tokenization without chat template.")
            formatted_prompt = prompt
            
        return formatted_prompt

    def generate_with_logprobs(
        self,
        prompts: List[str],
        temperature: float,
        max_tokens: int = 1,
        top_logprobs: int | None = 20,  # Number of top logprobs to return
        use_tqdm: bool = False,
    ) -> List[RequestOutput]:
        """
        Generate tokens with logprobs using vLLM.
        """
        if isinstance(top_logprobs, int):
            top_logprobs = min(top_logprobs, 20)  # vLLM supports up to 20 logprobs

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            # repetition_penalty=1.0,
            logprobs=top_logprobs,     # vllm only supports top_k logprobs self.model_config.max_logprobs = 20
            prompt_logprobs=None,  # Don't need prompt logprobs
            skip_special_tokens=False,
            # seed=self.seed,
        )
        
        outputs = self.llm.generate(
            prompts, 
            sampling_params, 
            use_tqdm=use_tqdm,
        )
        return outputs


class VLLMGenerator:
    def __init__(
        self, 
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.7,
        gpu_num: int | None = None,
        device: str = "cuda",
        seed: int = 55,
    ):
        self.generator = VLLMModelWrapper(
            model_name, 
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            gpu_num=gpu_num,
            device=device,
            seed=seed,
        )

    def default_generation(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_sequences: int = 10,
        max_new_tokens: int = 32_768,
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Generate multiple sequences using vLLM's batch processing.
        """
        formatted_prompt = self.generator.initial_tokenization(prompt)
        prompts = [formatted_prompt] * max_sequences
        
        outputs = self.generator.generate_with_logprobs(
            prompts, 
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_logprobs=20,
            use_tqdm=True,
        )

        generated_texts = []
        generated_entropies = []
        for output in outputs:
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
            sequence_entropies = [
                self.generator.compute_entropy_from_logprobs(
                    output.outputs[0].logprobs[i]
                )
                for i in range(len(output.outputs[0].logprobs))
            ]
            generated_entropies.append(sequence_entropies)
            
        return generated_texts, generated_entropies

    def entropy_aware_generation(
        self,
        prompt: str,
        temperature: float = 1.0,
        entropy_threshold: float = 2.5,
        dynamic_entropy: bool = True,
        initial_sequences: int = 1,
        max_sequences: int = 10,
        max_new_tokens: int = 64_000,
        max_tokens_without_branch: int = 1000,
        verbose: bool = True,
    ) -> Tuple[List[str], List[List[float]], List[dict[str, int | float]]]:
        """
        Generate sequences with entropy-aware branching using vLLM.
        Returns:
            - aware_generations: List of generated sequences up to max_sequences.
            - aware_entropies: List of lists containing entropies for each sequence.
        """
        assert initial_sequences > 0 and max_sequences >= initial_sequences, f"initial_sequences must be greater than 0 and max_sequences must be more than or equal to initial_sequences. {initial_sequences = }"

        formatted_prompt = self.generator.initial_tokenization(prompt)
        
        # Track active sequences and their entropies
        active_sequences = [formatted_prompt] * initial_sequences  # Start with initial sequences
        active_entropies = [[] * initial_sequences]  # Empty entropy history for each sequence
        completed_sequences = []
        completed_entropies = []
        # {
        #     "seq": 0,
        #     "step": 0,
        #     "token_id": 0,
        #     "entropy": 0.0,
        # }
        recorded_branches: List[dict[str, int | float]] = []
        
        start_time = time.time()
        token_step = 0
        dynamic_entropy_threshold = entropy_threshold  # Initialize dynamic threshold

        branching_phase = True          # True until the max_sequences is reached, slower inference
        tokens_since_last_branch = 0

        while active_sequences and token_step < max_new_tokens:
            # check if we reached the max_sequences limit
            total_sequences = len(active_sequences) + len(completed_sequences)
            if total_sequences >= max_sequences and branching_phase:
                if verbose:
                    print(f"\tReached max_sequences limit ({max_sequences}), fasten your seatbelts -> going to faster inference mode.")
                branching_phase = False

            if tokens_since_last_branch >= max_tokens_without_branch and branching_phase:
                if verbose:
                    print(f"\tReached {max_tokens_without_branch} tokens since last branching, switching to faster inference mode anyway.")
                branching_phase = False

            if branching_phase:
                # Entropy-aware generation with potential branching
                outputs = self.generator.generate_with_logprobs(
                    active_sequences, 
                    temperature=temperature,
                    max_tokens=1,
                    top_logprobs=20,       # should get a good approx of entropy
                    use_tqdm=False,
                )
                
                new_active_sequences = []
                new_sequence_entropies = []
                sequences_to_branch = []
                step_had_branching = False  # Track if any sequence branched in this step
                
                for i, output in enumerate(outputs):
                    if not output.outputs:
                        # Sequence finished
                        completed_sequences.append(active_sequences[i])
                        completed_entropies.append(active_entropies[i])
                        continue
                    
                    generated_output = output.outputs[0]
                    new_text = generated_output.text
                    
                    # Check if sequence is complete (contains EOS)
                    if self.generator.tokenizer.eos_token in new_text:
                        completed_sequences.append(active_sequences[i] + new_text)
                        completed_entropies.append(active_entropies[i] + [-9.0])      # EOS entropy
                        continue
                    
                    # Calculate entropy from logprobs
                    entropy = -1.0
                    if generated_output.logprobs:
                        # Get logprobs for the generated token
                        token_logprobs = generated_output.logprobs[0]  # First (and only) token
                        if token_logprobs:
                            entropy = self.generator.compute_entropy_from_logprobs(token_logprobs)
                    
                    # Update sequence
                    updated_sequence = active_sequences[i] + new_text
                    updated_entropies = active_entropies[i] + [entropy]
                    
                    # Calculate dynamic entropy threshold
                    if dynamic_entropy:
                        # Collect all valid entropies from all sequences
                        all_current_entropies = []
                        for seq_entropies in active_entropies + completed_entropies:
                            all_current_entropies.extend([e for e in seq_entropies if e >= 0])  # be sure to only include > 0 entropies, -1 stands for completed sequences
                        
                        if len(all_current_entropies) > 0:
                            # 99th percentile of current entropies
                            percentile_99 = np.percentile(all_current_entropies, 99)
                            dynamic_entropy_threshold = max(entropy_threshold, percentile_99)
                        else:
                            dynamic_entropy_threshold = entropy_threshold
                    else:
                        dynamic_entropy_threshold = entropy_threshold

                    # Check if we should branch
                    should_branch = (
                        entropy > dynamic_entropy_threshold and 
                        len(active_sequences) + len(new_active_sequences) + len(completed_sequences) < max_sequences and    # no more than max_sequences per prompt
                        "</think>" not in active_sequences[i][len(formatted_prompt):] + new_text    # Don't branch if thinking process ended
                    )

                    if should_branch:
                        step_had_branching = True
                        if verbose:
                            print(f"\tSequence {i} at step {token_step} reached entropy threshold: {entropy:.2f}")

                        top_2_token = sorted(
                            list(generated_output.logprobs[0].items()),
                            key=lambda x: x[1].rank
                        )

                        most_likely_token = self.generator.tokenizer.decode([top_2_token[0][0]])
                        second_most_likely_token = self.generator.tokenizer.decode([top_2_token[1][0]])

                        # overwrite the updated with the original sequence with most likely token
                        updated_sequence = active_sequences[i] + most_likely_token
                        updated_entropies = active_entropies[i] + [entropy]

                        # Create branching sequence with second most likely token
                        branching_sequence = active_sequences[i] + second_most_likely_token
                        branching_entropies = [-1] * len(active_entropies[i]) + [entropy]

                        sequences_to_branch.append((branching_sequence, branching_entropies))

                        # log branching
                        recorded_branches.append({
                            "seq_idx": i,                                       # current sequence index
                            "gen_step": token_step,                             # current generation step
                            "auto_selected_token": new_text,                    # the token chosen by the model
                            "entropy": entropy,                                 # its entropy
                            "entropy_threshold": float(dynamic_entropy_threshold),     # current dynamic entropy threshold
                            "man_token_original": most_likely_token,            # the token chosen by the user for seq_idx 
                            "man_token_branch": second_most_likely_token,       # the token chosen by the user for new_seq_idx
                            "new_seq_idx": len(active_sequences) + len(new_active_sequences) + len(completed_sequences),  # index of the new sequence
                            "last_generated_chars": active_sequences[i][-50:],  # last 50 chars of the sequence
                        })
                    
                    new_active_sequences.append(updated_sequence)
                    new_sequence_entropies.append(updated_entropies)

                # Update token counter based on whether any sequence branched in this step
                if step_had_branching:
                    tokens_since_last_branch = 0  # Reset counter when any sequence branches
                else:
                    tokens_since_last_branch += 1  # Increment counter only once per step
                
                # Add branched sequences
                for seq, entropies in sequences_to_branch:
                    if len(new_active_sequences) < max_sequences:
                        new_active_sequences.append(seq)
                        new_sequence_entropies.append(entropies.copy())     # Fresh entropy history for branches
                
                active_sequences = new_active_sequences
                active_entropies = new_sequence_entropies
                token_step += 1

            else:
                # Fast generation mode - complete all remaining sequences in batch
                remaining_tokens = max_new_tokens - token_step
                outputs = self.generator.generate_with_logprobs(
                    active_sequences, 
                    temperature=temperature,
                    max_tokens=remaining_tokens,
                    top_logprobs=None,  # No recording logprobs in fast mode
                    use_tqdm=True,
                )
                # Process completed sequences
                for i, output in enumerate(outputs):
                    if output.outputs:
                        generated_text = output.outputs[0].text
                        final_sequence = active_sequences[i] + generated_text

                        # not using logprobs here, placing -2 to indicate that branching is not active
                        final_entropies = active_entropies[i] + [-2.0] * len(generated_text.split())
                        
                        completed_sequences.append(final_sequence)
                        completed_entropies.append(final_entropies)
                    else:
                        # Empty output, just add the current sequence
                        completed_sequences.append(active_sequences[i])
                        completed_entropies.append(active_entropies[i])
                
                # Clear active sequences as all are now completed
                active_sequences = []
                active_entropies = []
                break
            
            if not active_sequences:
                print("\tAll sequences completed.")
                break
        
        # Combine active and completed sequences (active should be empty at this point)
        all_sequences = completed_sequences + active_sequences
        all_entropies = completed_entropies + active_entropies

        if verbose:
            # Calculate average entropy for completed sequences (only valid entropies)
            completed_entropies_flat = []
            for seq_entropies in completed_entropies:
                completed_entropies_flat.extend([e for e in seq_entropies if e >= 0])
            
            avg_completed_entropy = np.mean(completed_entropies_flat) if completed_entropies_flat else 0.0
            median_completed_entropy = np.median(completed_entropies_flat) if completed_entropies_flat else 0.0
            
            # Calculate 99th percentile of all entropies
            all_entropies_flat = []
            for seq_entropies in all_entropies:
                all_entropies_flat.extend([e for e in seq_entropies if e >= 0])
            
            percentile_99 = np.percentile(all_entropies_flat, 99) if all_entropies_flat else 0.0
            
            # Recap
            print(f"\tGenerated {len(completed_sequences)} completed sequences in {time.time() - start_time:.2f}s")
            print(f"\tAverage entropy for completed sequences: {avg_completed_entropy:.2f}")
            print(f"\tMedian entropy for completed sequences: {median_completed_entropy:.2f}")
            print(f"\t99th percentile of all entropies: {percentile_99:.2f}")
            print(f"\tLatest dynamic entropy threshold: {dynamic_entropy_threshold:.2f} (starting point was {entropy_threshold:.2f})")
        
        return all_sequences, all_entropies, recorded_branches


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = VLLMGenerator(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tensor_parallel_size=1,  # Increase if you have multiple GPUs
        max_model_len=65_536,
    )
    
    prompt = "What is the capital of France?"
    
    print("=== Default Generation ===")
    sequences = generator.default_generation(
        prompt=prompt,
        temperature=0.8,
        max_sequences=5,
        max_new_tokens=100
    )
    
    for i, seq in enumerate(sequences):
        print(f"Sequence {i+1}: {seq[:100]}...")
    
    print("\n=== Entropy-Aware Generation ===")
    sequences, entropies = generator.entropy_aware_generation(
        prompt=prompt,
        temperature=0.8,
        entropy_threshold=4.0,
        max_sequences=8,
        max_new_tokens=100,
        verbose=True
    )
    
    for i, (seq, seq_entropies) in enumerate(zip(sequences, entropies)):
        avg_entropy = np.mean([e for e in seq_entropies if e >= 0]) if any(e >= 0 for e in seq_entropies) else -1
        print(f"Sequence {i+1} (avg entropy: {avg_entropy:.2f}): {seq[:100]}...")
    
    print("\n=== Batch Complete Generation (Most Efficient) ===")
    sequences = generator.batch_complete_generation(
        prompt=prompt,
        temperature=0.8,
        max_sequences=5,
        max_new_tokens=100
    )
    
    for i, seq in enumerate(sequences):
        print(f"Sequence {i+1}: {seq[:100]}...")
