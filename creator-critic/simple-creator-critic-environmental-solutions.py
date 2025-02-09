import json
import argparse
import time
import re
import requests
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Configure model
DESIGNER_MODEL = "deepseek-r1:32b"
CRITIC_MODEL = "mistral-nemo"
SIMILARITY_THRESHOLD = 0.8

# Initialize the sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def query_ollama(prompt, temperature=0.6, is_critic=False):
    """Sends a request to Ollama's API."""
    payload = {
        "model": CRITIC_MODEL if is_critic else DESIGNER_MODEL,
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

def generate_initial_solution():
    """Generates an initial analysis of high-impact environmental challenges and solutions."""
    prompt = """Act as an environmental strategist focused on identifying and developing high-impact environmental solutions.

# Core Rules
- Focus on problems with global scale impact
- Prioritize urgency and time-sensitivity
- Consider implementation feasibility

# Your Task
Generate a comprehensive analysis of a high-impact environmental solution that includes:

- **Problem Analysis:**
  - What is the critical environmental challenge?
  - What is its quantifiable impact?
  
- **Solution Framework:**
  - What is the proposed solution?
  - What is its potential impact scale?
  
- **Impact Assessment:**
  - What are the specific environmental benefits?
  - When and how will we see results?

Focus on identifying and analyzing environmental challenges that, if addressed, would create the most significant positive impact."""

    return query_ollama(prompt)

def impact_critique(solution):
    """Reviews the solution focusing solely on the scale and significance of the challenge."""
    prompt = f"""Act as an environmental scientist specializing in global environmental impact assessment.
    
Evaluate the following environmental solution proposal.

Proposed Solution:
{solution}

Provide specific feedback in one or two sentences on:
- The scale and significance of the identified environmental challenge"""

    return query_ollama(prompt, is_critic=True)

def feasibility_critique(solution):
    """Reviews the solution focusing solely on implementation practicality."""
    prompt = f"""Act as an environmental policy expert with experience in large-scale environmental initiatives.
    
Evaluate the following environmental solution.

Proposed Solution:
{solution}

Provide specific feedback in one or two sentences on:
- The practicality of implementing this solution given current global conditions"""

    return query_ollama(prompt, is_critic=True)

def compute_similarity(text1, text2):
    """Compute semantic similarity between two texts."""
    # Get embeddings
    embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def iterate_solution(previous_solution, impact_feedback, feasibility_feedback, max_attempts=3):
    """Updates the solution based on the critique with similarity checking."""
    base_prompt = f"""Act as an environmental strategist focused on identifying and developing high-impact environmental solutions.

# Core Rules
- Focus on problems with global scale impact
- Prioritize urgency and time-sensitivity
- Consider implementation feasibility

# Previous Solution:
{previous_solution}

# Expert Feedback:
- **Impact Assessment:** {impact_feedback}
- **Implementation Feasibility:** {feasibility_feedback}

# Your Task
Refine the environmental solution while considering the feedback provided. Your analysis should include:

- **Problem Analysis:**
  - What is the critical environmental challenge?
  - What is its quantifiable impact?
  
- **Solution Framework:**
  - What is the proposed solution?
  - What is its potential impact scale?
  
- **Impact Assessment:**
  - What are the specific environmental benefits?
  - When and how will we see results?

Focus on identifying and analyzing environmental challenges that, if addressed, would create the most significant positive impact.

Each iteration must maintain sufficient similarity with the previous solution while incorporating the feedback."""

    for attempt in range(max_attempts):
        # Add increasing emphasis on similarity for each retry
        retry_emphasis = f"\n\nIMPORTANT: Your response MUST maintain at least {SIMILARITY_THRESHOLD*100}% semantic similarity with the previous solution. This is attempt {attempt + 1} of {max_attempts}." if attempt > 0 else ""
        
        response = query_ollama(base_prompt + retry_emphasis)
        
        # Clean response if it contains think blocks
        new_solution = response.split("</think>")[1].strip() if "</think>" in response else response
        
        # Check similarity
        similarity = compute_similarity(previous_solution, new_solution)
        print(f"Attempt {attempt + 1}: Similarity score = {similarity:.3f}")
        
        if similarity >= SIMILARITY_THRESHOLD:
            return response
            
    # If we've exhausted all attempts, return the previous solution and let it try again
    print(f"Warning: Could not achieve desired similarity after {max_attempts} attempts.")
    return previous_solution

def extract_think_block(response):
    """Extracts content inside the <think></think> block from the response."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else ""

def run_iteration_cycle(num_iterations=5):
    """Runs the iterative solution development process for a specified number of iterations."""
    results = {
        "iterations": [],
        "iteration_time": []
    }
    
    # Initial solution
    start_time = time.time()
    solution_response = generate_initial_solution()
    solution = solution_response.split("</think>")[1].strip() if "</think>" in solution_response else solution_response
    thoughts = extract_think_block(solution_response)
    end_time = time.time()
    results["iterations"].append({
        "iteration": 0,
        "solution": solution,
        "thoughts": thoughts,
        "impact_critique": None,
        "feasibility_critique": None,
    })
    results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
    
    # Iterative improvement cycle
    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} ===")
        start_time = time.time()
        
        # Get critique
        impact_critique_response = impact_critique(solution)
        print(f"Impact Critique: {impact_critique_response}")
        feasibility_critique_response = feasibility_critique(solution)
        print(f"Feasibility Critique: {feasibility_critique_response}")
        
        # Generate improved solution
        solution_response = iterate_solution(solution, impact_critique_response, feasibility_critique_response)
        solution = solution_response.split("</think>")[1].strip() if "</think>" in solution_response else solution_response
        thoughts = extract_think_block(solution_response)
        
        results["iterations"].append({
            "iteration": i + 1,
            "solution": solution,
            "thoughts": thoughts,
            "impact_critique": impact_critique_response,
            "feasibility_critique": feasibility_critique_response,
        })
        
        end_time = time.time()
        results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
        print(f"=== time taken: {round((end_time - start_time)/ 60, 2)} minutes")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run iterative environmental solution development')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations to run')
    args = parser.parse_args()
    
    final_results = run_iteration_cycle(args.num_iterations)
    
    # Save results
    filename = f"creator_critic_environmental_solutions_results_{args.num_iterations}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nExperiment completed! Results saved to {filename}") 