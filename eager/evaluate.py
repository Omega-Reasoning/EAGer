import re
from collections import Counter

def extract_boxed_answer(text: str) -> str:
    """
    Simpler approach that works for most cases without deeply nested braces.
    Extracts content from the LAST \boxed{} occurrence in the string.
    """
    # Find the start of the LAST \boxed{
    start = text.rfind('\\boxed{')
    if start == -1:
        return '' 
    
    # Start counting braces after the opening brace of \boxed{
    start_content = start + 7  # Length of '\\boxed{'
    brace_count = 1
    i = start_content
    
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        return text[start_content:i-1]

    return ''

def extract_boxed_answer_old(text: str) -> str:
    """Extract the answer from \boxed{} format."""
    # Look for \boxed{...} pattern
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Return the last boxed answer
    return ""

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()     # Remove whitespace and convert to lowercase
    answer = answer.replace(' ', '')    # Remove common mathematical formatting
    return answer

def is_correct(generated_answer: str, target_answer: int) -> bool:
    """Check if generated answer matches target."""
    extracted = extract_boxed_answer(generated_answer)
    if not extracted:
        return False
    try:    # to convert to int for comparison
        generated_int = int(normalize_answer(extracted))
        return generated_int == target_answer
    except (ValueError, TypeError):
        # If conversion fails, do string comparison
        equal = normalize_answer(extracted) == normalize_answer(str(target_answer))
        equal_in = normalize_answer(str(target_answer)) in normalize_answer(extracted)

        return equal or equal_in

def compute_average_accuracy(generations: list[list[str]], targets: list[int]) -> float:
    """
    Compute average accuracy across all generations for each problem.
    For each problem, computes the fraction of generations that are correct,
    then averages across all problems.
    """
    total_accuracy = 0
    total_problems = len(generations)
    
    for gen_list, target in zip(generations, targets):
        if len(gen_list) > 0:
            correct_gens = [int(is_correct(gen, target)) for gen in gen_list]
            problem_accuracy = sum(correct_gens) / len(correct_gens)
            total_accuracy += problem_accuracy
        # If no generations, this problem contributes 0 to the average
        
    return total_accuracy / total_problems if total_problems > 0 else 0.0


def compute_pass_at_1(generations: list[list[str]], targets: list[int]) -> float:
    """Compute pass@1 - probability that at least one generation is correct."""
    correct = 0
    total = len(generations)
    
    for gen_list, target in zip(generations, targets):
        if len(gen_list) > 0:
            # Check if ANY generation is correct
            if any(is_correct(gen, target) for gen in gen_list):
                correct += 1
        
    return correct / total if total > 0 else 0.0

def compute_cons_at_max(generations: list[list[str]], targets: list[int]) -> float:
    """Compute cons@max (majority vote accuracy)."""
    correct = 0
    total = len(generations)
    
    for gen_list, target in zip(generations, targets):
        # Extract all answers and count votes
        answers = []
        for gen in gen_list:
            extracted = extract_boxed_answer(gen)
            if extracted:
                answers.append(normalize_answer(extracted))
        
        # If no valid answers, this problem is incorrect (count as wrong)
        if not answers:
            continue  # Or set a flag to count as incorrect
            
        # Get majority vote
        vote_counts = Counter(answers)
        majority_answer = vote_counts.most_common(1)[0][0]
        
        # Check if majority answer is correct
        try:
            majority_int = int(majority_answer)
            if majority_int == target:
                correct += 1
        except (ValueError, TypeError):
            if majority_answer == normalize_answer(str(target)):
                correct += 1
    
    return correct / total if total > 0 else 0.0




# Example usage and comparison:
if __name__ == "__main__":
    # Example data: 2 problems, each with 3 generations
    generations = [
        ["\\boxed{5}", "\\boxed{3}", "\\boxed{5}"],  # Problem 1: 2/3 correct
        ["\\boxed{10}", "\\boxed{12}", "\\boxed{8}"] # Problem 2: 0/3 correct (target is 7)
    ]
    targets = [5, 7]
    
    avg_acc = compute_average_accuracy(generations, targets)
    pass_at_1 = compute_pass_at_1(generations, targets)
    cons_at_max = compute_cons_at_max(generations, targets)
    
    print(f"Average Accuracy: {avg_acc:.3f}")  # (2/3 + 0/3) / 2 = 0.333
    print(f"Pass@1: {pass_at_1:.3f}")          # 1/2 = 0.500 (problem 1 has ≥1 correct)
    print(f"Cons@Max: {cons_at_max:.3f}")      # 1/2 = 0.500 (problem 1 majority vote is correct)
