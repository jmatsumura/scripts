import json
import os
import requests
import argparse
import time
import re

# Configure Ollama API
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DESIGNER_MODEL = "deepseek-r1:32b"
CRITIC_MODEL = "mistral-nemo"

def query_ollama(prompt, temperature=0.7, is_critic=False):
    """Sends a request to Ollama's API."""
    payload = {
        "model": CRITIC_MODEL if is_critic else DESIGNER_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(OLLAMA_ENDPOINT, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "No response")
    else:
        return f"Error: {response.status_code} - {response.text}"

def generate_initial_design():
    """Generates the initial game design."""
    prompt = """Act as a creative video game designer.

Create a fun and innovative 2D video game concept. Write in a clear, concise format covering:
- Core Concept
- Main Mechanic
- Basic Gameplay Loop
- Art Style
- Development Scope

Be specific and practical in your design."""

    return query_ollama(prompt)

def critique_design(design):
    """Reviews the game design and suggests one way to make it more fun."""
    prompt = f"""System Prompt:
Assume the role of a video game design critic who knows what makes a game fun.

Here is a game design:
{design}

Your Task:
Suggest one specific thing that would make this game design more fun. Be brief and concrete in your suggestion."""

    return query_ollama(prompt, is_critic=True)

def iterate_design(original_design, critique):
    """Updates the game design based on the critique."""
    prompt = f"""Act as a creative video game designer.

# Background    
You've received feedback on your previous video game design from a critic. 

# Original Design:
{original_design}

# Critic's Feedback:
{critique}

# Your Task
Provide a new video game design that combines the best of your original design and the critic's feedback. 
Make sure your response is a complete video game design description, not just the changes."""
    
    return query_ollama(prompt)

def extract_think_block(response):
    """Extracts content inside the <think></think> block from the response."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else ""

def run_iteration_cycle(num_iterations=5):
    """Runs the iterative design process for a specified number of iterations."""
    results = {
        "iterations": [],
        "iteration_time": []
    }
    
    # Initial design
    start_time = time.time()
    design_response = generate_initial_design()
    design = design_response.split("</think>")[1].strip() if "</think>" in design_response else design_response
    thoughts = extract_think_block(design_response)
    end_time = time.time()
    results["iterations"].append({
        "iteration": 0,
        "design": design,
        "thoughts": thoughts,
        "critique": None,
    })
    results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
    
    # Iterative improvement cycle
    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} ===")
        start_time = time.time()
        
        # Get critique
        print("\n--- Critique ---")
        critique_response = critique_design(design)
        critique = critique_response.split("</think>")[1].strip() if "</think>" in critique_response else critique_response
        print(critique)
        
        # Generate improved design
        print("\n--- Updated Design ---")
        design_response = iterate_design(design, critique)
        design = design_response.split("</think>")[1].strip() if "</think>" in design_response else design_response
        thoughts = extract_think_block(design_response)
        print(design)
        
        results["iterations"].append({
            "iteration": i + 1,
            "design": design,
            "thoughts": thoughts,
            "critique": critique,
        })
        
        end_time = time.time()
        results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
        print(f"@@@@@ time taken: {round((end_time - start_time)/ 60, 2)} minutes")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run iterative game design experiment')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations to run')
    args = parser.parse_args()
    
    final_results = run_iteration_cycle(args.num_iterations)
    
    # Save results
    filename = f"creator_critic_iterative_deepseek_results_{args.num_iterations}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nExperiment completed! Results saved to {filename}") 