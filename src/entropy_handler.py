import torch
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from tqdm import tqdm

from vllm import SamplingParams, EngineArgs, LLMEngine
from transformers import AutoTokenizer


class VLLMGeneratorEfficient:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.7,
        seed: int = 55,
    ):
        """Initialize the vLLM generator with specified parameters."""
        print(f"Loading model with vLLM: {model_name}")
        
        engine_args = EngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            trust_remote_code=True,
            enforce_eager=False,
            enable_prefix_caching=True,
        )
        
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.request_id_counter = 0
        self.model_name = model_name
        
        print(f"Model loaded successfully")

    def _get_next_request_id(self) -> str:
        """Generate a unique request ID for each generation request."""
        self.request_id_counter += 1
        return f"req_{self.request_id_counter}"

    def initial_tokenization(self, prompt: str) -> str:
        """Format prompt using chat template if available."""
        if self.tokenizer.chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            print("Warning: Using default tokenization without chat template.")
            formatted_prompt = prompt
            
        # if 'gpt' is in self.model_name.lower():
        #     formatted_prompt.replace("Reasoning: medium", "Reasoning: high")

        return formatted_prompt

    def compute_entropy_from_logprobs(self, token_logprobs: Dict[int, Any]) -> float:
        """Compute entropy from vLLM's logprobs output."""
        if not token_logprobs:
            return -1.0

        try:
            logprob_values = []
            for token_id, logprob_obj in token_logprobs.items():
                if hasattr(logprob_obj, 'logprob'):
                    logprob_values.append(logprob_obj.logprob)
                else:
                    logprob_values.append(float(logprob_obj))

            if not logprob_values:
                return -1.0

            logprobs_tensor = torch.tensor(logprob_values, dtype=torch.float32)
            probs = torch.softmax(logprobs_tensor, dim=0)
            
            # Calculate Shannon entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            return entropy

        except Exception as e:
            print(f"Warning: Error computing entropy: {e}")
            return -1.0

    def _should_branch(
        self, 
        entropy: float, 
        entropy_threshold: float, 
        sequence: str, 
        formatted_prompt: str, 
        current_sequences: int, 
        max_sequences: int,
    ) -> bool:
        """Check if branching conditions are met."""
        return (
            entropy > entropy_threshold and 
            current_sequences < max_sequences and
            "</think>" not in sequence[len(formatted_prompt):]
        )

    def _get_top_2_tokens(self, token_logprobs: Dict[int, Any]) -> Tuple[int, int, str, str]:
        """Get the top 2 most likely tokens with their IDs."""
        if not token_logprobs:
            return -1, -1, "", ""
        
        # Sort by rank (lower rank = more likely)
        sorted_tokens = sorted(
            token_logprobs.items(),
            key=lambda x: x[1].rank if hasattr(x[1], 'rank') else float('inf')
        )
        
        if len(sorted_tokens) < 2:
            return -1, -1, "", ""
        
        most_likely_id = sorted_tokens[0][0]
        second_most_likely_id = sorted_tokens[1][0]
        most_likely_token = self.tokenizer.decode([most_likely_id])
        second_most_likely_token = self.tokenizer.decode([second_most_likely_id])
        
        return most_likely_id, second_most_likely_id, most_likely_token, second_most_likely_token

    def _is_eos_token(self, token_id: int) -> bool:
        """Check if token is an EOS token."""
        return token_id == self.tokenizer.eos_token_id

    def _check_eos_in_text(self, text: str) -> bool:
        """Check if text contains EOS token."""
        return self.tokenizer.eos_token in text if self.tokenizer.eos_token else False

    def entropy_aware_generation(
        self,
        prompt: str,
        temperature: float = 1.0,
        entropy_threshold: float = 2.5,
        initial_sequences: int = 1,
        max_sequences: int = 10,
        max_new_tokens: int = 64_000,
        max_tokens_without_branch: int = 1000,
        verbose: bool = True,
    ) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """
        Generate sequences with entropy-aware branching using vLLM Engine API.
        Token-by-token generation with entropy checking after each token.
        
        Returns:
            - generated_sequences: List of generated sequences
            - sequence_entropies: List of lists containing entropies for each sequence
            - recorded_branches: List of dictionaries with branching information
        """
        assert initial_sequences > 0 and max_sequences >= initial_sequences, \
            f"initial_sequences must be > 0 and max_sequences >= initial_sequences"

        formatted_prompt = self.initial_tokenization(prompt)
        
        # Initialize tracking variables
        active_sequences = {}  # request_id -> current_text
        active_entropies = {}  # request_id -> list of entropies
        completed_sequences = []
        completed_entropies = []
        recorded_branches = []
        
        # Create initial sequences - each generates exactly 1 token
        for i in range(initial_sequences):
            request_id = self._get_next_request_id()
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=1,
                logprobs=20,
                ignore_eos=True,  # Ignore EOS so we can detect it ourselves
                include_stop_str_in_output=False
            )
            
            self.engine.add_request(request_id, formatted_prompt, sampling_params)
            active_sequences[request_id] = formatted_prompt
            active_entropies[request_id] = []
        
        start_time = time.time()
        token_step = 0
        branching_phase = True
        tokens_since_last_branch = 0
        
        if verbose:
            print(f"Starting entropy-aware generation with {initial_sequences} initial sequences")
        
        # Initialize progress tracking
        pbar = tqdm(total=max_new_tokens, desc="Generating tokens", unit="tok") if verbose else None
        last_update_time = time.time()
        last_token_count = 0
        
        while active_sequences and token_step < max_new_tokens:
            # Check if we should exit branching phase
            total_sequences = len(active_sequences) + len(completed_sequences)
            if total_sequences >= max_sequences and branching_phase:
                if verbose:
                    print(f"\nReached max_sequences limit ({max_sequences}), switching to fast mode")
                branching_phase = False
            
            if tokens_since_last_branch >= max_tokens_without_branch and branching_phase:
                if verbose:
                    print(f"\nReached {max_tokens_without_branch} tokens without branching, switching to fast mode")
                branching_phase = False
            
            # Update progress bar
            if pbar is not None:
                current_time = time.time()
                time_diff = current_time - last_update_time
                
                if time_diff >= 1.0:
                    tokens_generated = token_step - last_token_count
                    tokens_per_second = tokens_generated / time_diff if time_diff > 0 else 0
                    
                    pbar.set_description(
                        f"Step {token_step}/{max_new_tokens} | "
                        f"Active: {len(active_sequences)} | "
                        f"Completed: {len(completed_sequences)} | "
                        f"TPS: {tokens_per_second:.1f}"
                    )
                    
                    last_update_time = current_time
                    last_token_count = token_step
                
                pbar.update(1)
            
            # Process one generation step
            request_outputs = self.engine.step()
            
            new_sequences_to_add = []
            step_had_branching = False
            completed_in_step = []
            new_requests_to_add = []
            
            for request_output in request_outputs:
                if request_output.request_id not in active_sequences:
                    continue
                
                current_sequence = active_sequences[request_output.request_id]
                current_entropies = active_entropies[request_output.request_id]
                
                # Check if we got output
                if not request_output.outputs:
                    # No output, mark as completed
                    completed_sequences.append(current_sequence)
                    completed_entropies.append(current_entropies)
                    completed_in_step.append(request_output.request_id)
                    continue
                
                generated_output = request_output.outputs[0]
                new_text = generated_output.text
                
                # Check if we got an empty result
                if not new_text:
                    completed_sequences.append(current_sequence)
                    completed_entropies.append(current_entropies)
                    completed_in_step.append(request_output.request_id)
                    continue
                
                # Check for EOS token in the generated text
                if self._check_eos_in_text(new_text):
                    # Sequence completed with EOS
                    completed_sequences.append(new_text)
                    completed_entropies.append(current_entropies + [-9.0])  # Special entropy for EOS
                    completed_in_step.append(request_output.request_id)
                    continue
                
                # Calculate entropy from logprobs
                entropy = -1.0
                if generated_output.logprobs and generated_output.logprobs[0]:
                    entropy = self.compute_entropy_from_logprobs(generated_output.logprobs[0])
                
                # Update sequence and entropies
                updated_sequence = new_text
                updated_entropies = current_entropies + [entropy]
                
                # Check if we should branch (only in branching phase)
                should_branch = False
                if branching_phase and entropy > 0:
                    should_branch = self._should_branch(
                        entropy, entropy_threshold, updated_sequence, 
                        formatted_prompt, len(active_sequences) + len(new_sequences_to_add), 
                        max_sequences
                    )
                
                if should_branch and generated_output.logprobs and generated_output.logprobs[0]:
                    step_had_branching = True
                    if verbose:
                        seq_idx = list(active_sequences.keys()).index(request_output.request_id)
                        print(f"Branching: sequence {seq_idx} at step {token_step}, entropy: {entropy:.2f}")
                    
                    # Get top 2 tokens
                    most_likely_id, second_most_likely_id, most_likely_token, second_most_likely_token = self._get_top_2_tokens(
                        generated_output.logprobs[0]
                    )
                    
                    if most_likely_id != -1 and second_most_likely_id != -1:
                        # Create primary sequence (with most likely token)
                        primary_sequence = current_sequence + most_likely_token
                        
                        # Create branch sequence (with second most likely token)
                        branch_sequence = current_sequence + second_most_likely_token
                        
                        # Record branching information
                        seq_idx = list(active_sequences.keys()).index(request_output.request_id)
                        recorded_branches.append({
                            "seq_idx": seq_idx,
                            "gen_step": token_step,
                            "auto_selected_token": new_text,
                            "entropy": entropy,
                            "entropy_threshold": entropy_threshold,
                            "primary_token": most_likely_token,
                            "branch_token": second_most_likely_token,
                            "new_seq_idx": len(active_sequences) + len(new_sequences_to_add),
                            "context": current_sequence[-50:],
                        })
                        
                        # Update current sequence with primary token
                        active_sequences[request_output.request_id] = primary_sequence
                        active_entropies[request_output.request_id] = updated_entropies
                        
                        # Add new branch sequence
                        if len(active_sequences) + len(new_sequences_to_add) < max_sequences:
                            new_sequences_to_add.append({
                                'sequence': branch_sequence,
                                'entropies': updated_entropies,
                            })
                        
                        # Create new request for continuing the primary sequence
                        if not branching_phase:
                            # In fast mode, use higher max_tokens
                            remaining_tokens = max_new_tokens - token_step
                            new_sampling_params = SamplingParams(
                                temperature=temperature,
                                max_tokens=remaining_tokens,
                                logprobs=None,
                                ignore_eos=False
                            )
                        else:
                            # In branching mode, continue token by token
                            new_sampling_params = SamplingParams(
                                temperature=temperature,
                                max_tokens=1,
                                logprobs=20,
                                ignore_eos=True
                            )
                        
                        new_requests_to_add.append({
                            'request_id': request_output.request_id,
                            'sequence': primary_sequence,
                            'sampling_params': new_sampling_params
                        })
                        
                    else:
                        # Fallback: just update without branching
                        active_sequences[request_output.request_id] = updated_sequence
                        active_entropies[request_output.request_id] = updated_entropies
                        
                        # Continue with next token
                        if not branching_phase:
                            remaining_tokens = max_new_tokens - token_step
                            new_sampling_params = SamplingParams(
                                temperature=temperature,
                                max_tokens=remaining_tokens,
                                logprobs=None,
                                ignore_eos=False
                            )
                        else:
                            new_sampling_params = SamplingParams(
                                temperature=temperature,
                                max_tokens=1,
                                logprobs=20,
                                ignore_eos=True
                            )
                        
                        new_requests_to_add.append({
                            'request_id': request_output.request_id,
                            'sequence': updated_sequence,
                            'sampling_params': new_sampling_params
                        })
                else:
                    # No branching, just update and continue
                    active_sequences[request_output.request_id] = updated_sequence
                    active_entropies[request_output.request_id] = updated_entropies
                    
                    # Continue with next token
                    if not branching_phase:
                        remaining_tokens = max_new_tokens - token_step
                        new_sampling_params = SamplingParams(
                            temperature=temperature,
                            max_tokens=remaining_tokens,
                            logprobs=None,
                            ignore_eos=False
                        )
                    else:
                        new_sampling_params = SamplingParams(
                            temperature=temperature,
                            max_tokens=1,
                            logprobs=20,
                            ignore_eos=True
                        )
                    
                    new_requests_to_add.append({
                        'request_id': request_output.request_id,
                        'sequence': updated_sequence,
                        'sampling_params': new_sampling_params
                    })
            
            # Remove completed sequences
            for request_id in completed_in_step:
                if request_id in active_sequences:
                    del active_sequences[request_id]
                    del active_entropies[request_id]
            
            # Add new branch sequences
            for new_seq_data in new_sequences_to_add:
                if len(active_sequences) < max_sequences:
                    new_request_id = self._get_next_request_id()
                    
                    if not branching_phase:
                        remaining_tokens = max_new_tokens - token_step
                        sampling_params = SamplingParams(
                            temperature=temperature,
                            max_tokens=remaining_tokens,
                            logprobs=None,
                            ignore_eos=False
                        )
                    else:
                        sampling_params = SamplingParams(
                            temperature=temperature,
                            max_tokens=1,
                            logprobs=20,
                            ignore_eos=True
                        )
                    
                    self.engine.add_request(new_request_id, new_seq_data['sequence'], sampling_params)
                    active_sequences[new_request_id] = new_seq_data['sequence']
                    active_entropies[new_request_id] = new_seq_data['entropies']
            
            # Add continuation requests
            for req_data in new_requests_to_add:
                self.engine.add_request(req_data['request_id'], req_data['sequence'], req_data['sampling_params'])
            
            # Update counters
            if step_had_branching:
                tokens_since_last_branch = 0
            else:
                tokens_since_last_branch += 1
            
            token_step += 1
            
            # In fast mode, let sequences complete naturally
            if not branching_phase:
                # Process remaining requests until completion
                while active_sequences:
                    request_outputs = self.engine.step()
                    
                    for request_output in request_outputs:
                        if request_output.request_id in active_sequences:
                            if request_output.finished:
                                current_entropies = active_entropies[request_output.request_id]
                                
                                if request_output.outputs:
                                    final_text = request_output.outputs[0].text
                                    # For fast mode, use -2.0 as entropy indicator
                                    final_entropies = current_entropies + [-2.0] * (len(final_text.split()) - len(current_entropies))
                                else:
                                    final_text = active_sequences[request_output.request_id]
                                    final_entropies = current_entropies
                                
                                completed_sequences.append(final_text)
                                completed_entropies.append(final_entropies)
                                
                                # Remove from active
                                del active_sequences[request_output.request_id]
                                del active_entropies[request_output.request_id]
                
                break  # Exit main loop after fast mode completion
        
        # Close progress bar
        if pbar is not None:
            pbar.close()
        
        # Combine remaining active sequences with completed ones
        all_sequences = completed_sequences + list(active_sequences.values())
        all_entropies = completed_entropies + list(active_entropies.values())
        
        if verbose:
            # Calculate and print statistics
            total_entropies = []
            for seq_entropies in all_entropies:
                total_entropies.extend([e for e in seq_entropies if e >= 0])
            
            avg_entropy = np.mean(total_entropies) if total_entropies else 0.0
            median_entropy = np.median(total_entropies) if total_entropies else 0.0
            
            elapsed_time = time.time() - start_time
            print(f"Generation completed in {elapsed_time:.2f}s")
            print(f"Generated {len(all_sequences)} sequences with {len(recorded_branches)} branches")
            print(f"Average entropy: {avg_entropy:.2f}, Median entropy: {median_entropy:.2f}")
            print(f"Entropy threshold: {entropy_threshold:.2f}")
        
        return all_sequences, all_entropies, recorded_branches

    def default_generation(
        self, 
        prompt: str, 
        temperature: float = 1.0, 
        max_sequences: int = 10, 
        max_new_tokens: int = 32_768,
        verbose: bool = True,
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Generate multiple sequences using vLLM's batch processing.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_sequences: Number of sequences to generate
            max_new_tokens: Maximum tokens per sequence
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (generated_texts, generated_entropies)
        """
        formatted_prompt = self.initial_tokenization(prompt)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            logprobs=20,
            ignore_eos=False
        )
        
        # Add all requests to the engine
        request_ids = []
        for i in range(max_sequences):
            request_id = self._get_next_request_id()
            self.engine.add_request(request_id, formatted_prompt, sampling_params)
            request_ids.append(request_id)
        
        # Track completed outputs
        completed_outputs = {}
        
        if verbose:
            print(f"Starting batch generation of {max_sequences} sequences...")
            start_time = time.time()
        
        # Process until all requests are complete
        while len(completed_outputs) < max_sequences:
            request_outputs = self.engine.step()
            
            for request_output in request_outputs:
                if request_output.request_id in request_ids and request_output.finished:
                    completed_outputs[request_output.request_id] = request_output
                    
                    if verbose and len(completed_outputs) % max(1, max_sequences // 10) == 0:
                        print(f"Completed {len(completed_outputs)}/{max_sequences} sequences...")
        
        # Extract results in order
        generated_texts = []
        generated_entropies = []
        
        for request_id in request_ids:
            if request_id in completed_outputs:
                output = completed_outputs[request_id]
                
                if output.outputs:
                    generated_text = output.outputs[0].text
                    generated_texts.append(generated_text)
                    
                    # Calculate entropies for each token
                    sequence_entropies = []
                    if output.outputs[0].logprobs:
                        for token_logprobs in output.outputs[0].logprobs:
                            if token_logprobs:
                                entropy = self.compute_entropy_from_logprobs(token_logprobs)
                                sequence_entropies.append(entropy)
                            else:
                                sequence_entropies.append(-1.0)
                    
                    generated_entropies.append(sequence_entropies)
                else:
                    generated_texts.append("")
                    generated_entropies.append([])
            else:
                generated_texts.append("")
                generated_entropies.append([])
        
        if verbose:
            elapsed_time = time.time() - start_time
            total_tokens = sum(len(entropies) for entropies in generated_entropies)
            avg_tokens_per_sequence = total_tokens / len(generated_texts) if generated_texts else 0
            
            all_entropies_flat = []
            for seq_entropies in generated_entropies:
                all_entropies_flat.extend([e for e in seq_entropies if e >= 0])
            
            avg_entropy = np.mean(all_entropies_flat) if all_entropies_flat else 0.0
            median_entropy = np.median(all_entropies_flat) if all_entropies_flat else 0.0
            
            print(f"Batch generation completed in {elapsed_time:.2f}s")
            print(f"Generated {len(generated_texts)} sequences, avg {avg_tokens_per_sequence:.1f} tokens/seq")
            print(f"Average entropy: {avg_entropy:.2f}, Median entropy: {median_entropy:.2f}")
        
        return generated_texts, generated_entropies
