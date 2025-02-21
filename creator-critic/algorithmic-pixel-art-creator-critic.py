import json
import argparse
import time
import requests
import os
import re
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import random

# Configure models
CREATOR_MODEL = os.getenv("CREATOR_MODEL", "deepseek-r1:32b")  # For algorithm generation
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "qwen2.5:32b")      # For algorithm refinement
SIMILARITY_THRESHOLD = 0.7  # Higher threshold for uniqueness

# Initialize sentence transformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def query_ollama(prompt, temperature=0.6, is_critic=False):
    """Sends a request to Ollama's API."""
    model = CRITIC_MODEL if is_critic else CREATOR_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    if response.status_code == 200:
        return response.json().get("response", "No response")
    else:
        return f"Error: {response.status_code} - {response.text}"

def generate_idea():
    """Generate a single algorithmic approach to sword pixel art."""
    prompt = """Act as a generative art engineer obsessed with creating novel systems for procedural pixel art.

Design an algorithm that can draw a sword in a 16x16 or 32x32 pixel grid.

# Key Considerations
1. What are the fundamental components of a sword?
2. How can we break this down into clear algorithmic steps?
3. What novel approach could make this more viable?

# Your Response Should Include
— The algorithm you propose"""

    return query_ollama(prompt=prompt, temperature=0.6)

def compute_similarity(text1, text2):
    """Compute semantic similarity between two texts."""
    embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
    similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def is_unique_insight(insight, existing_insights):
    """Check if an insight is semantically unique compared to existing ones."""
    if not existing_insights:
        return True
    
    for existing in existing_insights:
        similarity = compute_similarity(insight, existing)
        if similarity > SIMILARITY_THRESHOLD:
            return False
    return True

def extract_key_thought(idea):
    """Extract the most valuable thought/insight from an idea."""
    prompt = f"""Act as an insight curator focused on novel problem-solving approaches.

Review this approach to pixel art sword generation:

{idea}

# Task
Extract the SINGLE most valuable insight about how to think about or approach this problem.
Focus on the thought process, not the technical implementation.
Be very specific and concrete.

Respond with just the insight in a sentence or two."""

    return query_ollama(prompt=prompt, temperature=0.3, is_critic=True)

def extract_key_design(idea):
    """Extract the most valuable technical/design insight from an idea."""
    prompt = f"""Act as a technical curator focused on algorithmic innovations.

Review this approach to pixel art sword generation:

{idea}

# Task
Extract the SINGLE most valuable technical or design insight from this approach.
Focus on the implementation details, not the general thinking.
Be very specific and concrete.

Respond with just the technical insight in a sentence or two."""

    return query_ollama(prompt=prompt, temperature=0.3, is_critic=True)

def generate_iteration(previous_idea):
    """Generate an iteration building on a previous idea."""
    prompt = f"""Act as a generative art engineer obsessed with creating novel systems for procedural pixel art.

Here's a previous approach to drawing a sword in a pixel grid:

{previous_idea}

# Task
Build on this idea in ONE specific way. Pick something interesting from the approach and evolve it further.
Don't try to change everything - focus on making one aspect significantly better.

# Your Response Should Include
— Your evolved algorithm"""

    return query_ollama(prompt=prompt, temperature=0.6)

def explore_algorithms(num_ideas=100):
    """Generate many ideas and collect unique valuable insights, alternating between new ideas and iterations."""
    results = {
        "unique_thoughts": [],
        "unique_designs": [],
        "creation_time": []
    }
    
    # Track our current working ideas
    current_ideas = []
    current_algorithms = []
    
    # Generate ideas and extract insights
    print(f"\n=== Generating Ideas and Extracting Insights ===")
    start_time = time.time()
    iteration_start = time.time()
    
    for i in range(num_ideas):
        print(f"\nProcessing iteration {i + 1}/{num_ideas}")
        
        # Alternate between new ideas and iterations
        if i % 2 == 0 or not current_ideas:
            # Generate completely new idea
            print("Generating new idea...")
            response = generate_idea()
            # Split into thinking and algorithm parts
            thinking = extract_think_block(response)
            algorithm = response.split("</think>")[-1].strip() if "</think>" in response else response
            current_ideas.append(thinking)
            current_algorithms.append(algorithm)
        else:
            # Pick a random previous idea to iterate on
            print("Iterating on previous idea...")
            idx = random.randrange(len(current_algorithms))
            previous_idea = current_algorithms[idx]
            response = generate_iteration(previous_idea)
            # Split into thinking and algorithm parts
            thinking = extract_think_block(response)
            algorithm = response.split("</think>")[-1].strip() if "</think>" in response else response
            current_ideas.append(thinking)
            current_algorithms.append(algorithm)
        
        # Extract and check thought insight
        thought = extract_key_thought(thinking if thinking else algorithm)
        if is_unique_insight(thought, results["unique_thoughts"]):
            print("Found unique thought insight!")
            results["unique_thoughts"].append(thought)
        
        # Extract and check design insight
        design = extract_key_design(algorithm)
        if is_unique_insight(design, results["unique_designs"]):
            print("Found unique design insight!")
            results["unique_designs"].append(design)
        
        # Keep only the last 5 ideas for iteration
        if len(current_ideas) > 5:
            current_ideas = current_ideas[-5:]
            current_algorithms = current_algorithms[-5:]
        
        # Print time taken for this iteration
        iteration_end = time.time()
        print(f"Iteration time: {round((iteration_end - iteration_start) / 60, 2) } minutes")
        iteration_start = time.time()
    
    generation_time = time.time() - start_time
    results["creation_time"].append(round(generation_time / 60, 2))
    
    return results

def extract_think_block(response):
    """Extracts content inside the <think></think> block from the response."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else ""

def reprocess_insights_with_threshold(insights, new_threshold):
    """Reprocess a list of insights with a higher similarity threshold to remove near-duplicates.
    Keeps insights that are LESS similar than the threshold to each other."""
    unique_insights = []
    
    for insight in insights:
        if not unique_insights:
            unique_insights.append(insight)
            continue
            
        is_unique = True
        for existing in unique_insights:
            similarity = compute_similarity(insight, existing)
            if similarity >= new_threshold:  # If similarity is higher than threshold, it's too similar
                is_unique = False
                break
        
        if is_unique:
            unique_insights.append(insight)
    
    return unique_insights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Explore sword pixel art algorithm approaches')
    parser.add_argument('--num_ideas', type=int, default=100, help='Number of ideas to generate (default: 100)')
    parser.add_argument('--reprocess_threshold', type=float, help='Remove insights with similarity HIGHER than this threshold (higher value = more permissive, allows more similar items)')
    parser.add_argument('--input_file', help='Input JSON file to reprocess (if not specified, will generate new results)')
    args = parser.parse_args()
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if args.input_file:
        # Load existing results
        print(f"\nLoading results from {args.input_file}")
        with open(args.input_file, 'r') as f:
            final_results = json.load(f)
    else:
        # Generate new results
        final_results = explore_algorithms(args.num_ideas)
    
    # Reprocess with higher threshold if specified
    if args.reprocess_threshold is not None:
        print(f"\nReprocessing insights with similarity threshold: {args.reprocess_threshold}")
        original_thoughts = final_results["unique_thoughts"]
        original_designs = final_results["unique_designs"]
        
        final_results["unique_thoughts"] = reprocess_insights_with_threshold(original_thoughts, args.reprocess_threshold)
        final_results["unique_designs"] = reprocess_insights_with_threshold(original_designs, args.reprocess_threshold)
        
        print(f"\nAfter reprocessing:")
        print(f"Thoughts reduced from {len(original_thoughts)} to {len(final_results['unique_thoughts'])}")
        print(f"Designs reduced from {len(original_designs)} to {len(final_results['unique_designs'])}")
        
        # Only save to a new file if we did reprocessing
        filename = f"creator_critic_sword_algorithm_exploration_reprocessed_{timestamp}_results.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
        print(f"\nReprocessed results saved to: {filename}")
    elif not args.input_file:
        # Only save if we generated new results and didn't reprocess
        filename = f"creator_critic_sword_algorithm_exploration_{timestamp}_results.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
        print(f"\nResults saved to: {filename}")
    
    # Print insights
    print("\nUnique Thought Insights:", len(final_results["unique_thoughts"]))
    for t in final_results["unique_thoughts"]:
        print(f"- {t}")
    
    print("\nUnique Design Insights:", len(final_results["unique_designs"]))
    for d in final_results["unique_designs"]:
        print(f"- {d}")
    
    if "creation_time" in final_results:
        print("\nTime Breakdown (minutes):")
        print(f"Idea Generation & Insight Extraction: {final_results['creation_time'][0]}") 