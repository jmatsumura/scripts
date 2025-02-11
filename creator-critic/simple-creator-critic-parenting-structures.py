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

def generate_initial_structure():
    """Generates an initial analysis of balanced parenting structures."""
    prompt = """Act as a child development expert focused on balancing opportunities and risks for children.

# Core Question
For an 8-year-old in a modern household in the United States, what structure provides the optimal balance between enabling growth opportunities and ensuring safety?

# Your Task
Analyze how to structure a child's environment to:

- **Safety Framework:**
  - What boundaries are essential for safety?
  - Which risks have developmental value and how should they be managed?

- **Growth Opportunities:**
  - Which physical challenges should be encouraged? (sports, playground)
  - Which social challenges should be supported? (friendships, conflicts)

Focus on practical guidelines that maximize learning while maintaining appropriate safety."""

    return query_ollama(prompt)

def child_perspective_critique(structure):
    """Reviews the structure from a child's perspective."""
    prompt = f"""Act as a precocious and independent-minded 8-year-old who is highly capable and often questions adult rules.
    You're smart enough to understand safety concerns, but you're also frustrated when rules seem unnecessary or limiting.
    You have strong opinions about your abilities and often feel adults underestimate what you can handle.
    
Evaluate the following parenting structure from your perspective:

Structure:
{structure}

Provide specific feedback in one or two sentences on:
- How these rules might unnecessarily limit your capabilities or how they could better respect your judgment while keeping you safe"""

    return query_ollama(prompt, is_critic=True)

def primary_parent_critique(structure):
    """Reviews the structure from a primary parent's perspective, focusing on daily implementation."""
    prompt = f"""Act as a primary parent who lives with and implements these structures daily.
    Your focus is on practical implementation and maintaining family harmony.
    
Evaluate the following parenting structure:

Structure:
{structure}

Provide specific feedback in one or two sentences on:
- How feasible this structure is to implement in daily family life while maintaining household harmony"""

    return query_ollama(prompt, is_critic=True)

def mentor_critique(structure):
    """Reviews the structure from a mentor's perspective, focusing on societal integration."""
    prompt = f"""Act as an experienced mentor (like a teacher, coach, or counselor) who regularly works
    with many different children and families across various contexts and backgrounds.
    Your focus is on how this structure prepares children for broader social contexts.
    
Evaluate the following parenting structure:

Structure:
{structure}

Provide specific feedback in one or two sentences on:
- How effectively this structure prepares children for success across different social contexts (school, activities, peer groups)"""

    return query_ollama(prompt, is_critic=True)

def compute_similarity(text1, text2):
    """Compute semantic similarity between two texts."""
    # Get embeddings
    embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def iterate_structure(previous_structure, child_feedback, parent_feedback, mentor_feedback, max_attempts=3):
    """Updates the structure based on the critique with similarity checking."""
    base_prompt = f"""Act as a child development expert focused on balancing opportunities and risks for children.

# Core Question
For an 8-year-old in a modern household in the United States, what structure provides the optimal balance between enabling growth opportunities and ensuring safety?

# Previous Structure:
{previous_structure}

# Expert Feedback:
- **Child's Perspective:** {child_feedback}
- **Parent's Perspective:** {parent_feedback}
- **Mentor's Perspective:** {mentor_feedback}

# Your Task
Analyze how to structure a child's environment to:

- **Safety Framework:**
  - What boundaries are essential for safety?
  - Which risks have developmental value and how should they be managed?

- **Growth Opportunities:**
  - Which physical challenges should be encouraged? (sports, playground)
  - Which social challenges should be supported? (friendships, conflicts)

Focus on practical guidelines that maximize learning while maintaining appropriate safety.

Each iteration must maintain sufficient similarity with the previous structure while incorporating the feedback."""

    for attempt in range(max_attempts):
        # Add increasing emphasis on similarity for each retry
        retry_emphasis = f"\n\nIMPORTANT: Your response MUST maintain at least {SIMILARITY_THRESHOLD*100}% semantic similarity with the previous structure. This is attempt {attempt + 1} of {max_attempts}." if attempt > 0 else ""
        
        response = query_ollama(base_prompt + retry_emphasis)
        
        # Clean response if it contains think blocks
        new_structure = response.split("</think>")[1].strip() if "</think>" in response else response
        
        # Check similarity
        similarity = compute_similarity(previous_structure, new_structure)
        print(f"Attempt {attempt + 1}: Similarity score = {similarity:.3f}")
        
        if similarity >= SIMILARITY_THRESHOLD:
            return response
            
    # If we've exhausted all attempts, return the previous structure and let it try again
    print(f"Warning: Could not achieve desired similarity after {max_attempts} attempts.")
    return previous_structure

def extract_think_block(response):
    """Extracts content inside the <think></think> block from the response."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else ""

def run_iteration_cycle(num_iterations=5):
    """Runs the iterative structure development process for a specified number of iterations."""
    results = {
        "iterations": [],
        "iteration_time": []
    }
    
    # Initial structure
    start_time = time.time()
    structure_response = generate_initial_structure()
    structure = structure_response.split("</think>")[1].strip() if "</think>" in structure_response else structure_response
    thoughts = extract_think_block(structure_response)
    end_time = time.time()
    results["iterations"].append({
        "iteration": 0,
        "structure": structure,
        "thoughts": thoughts,
        "child_critique": None,
        "parent_critique": None,
        "mentor_critique": None
    })
    results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
    
    # Iterative improvement cycle
    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} ===")
        start_time = time.time()
        
        # Get critiques
        child_critique_response = child_perspective_critique(structure)
        print(f"Child's Perspective: {child_critique_response}")
        parent_critique_response = primary_parent_critique(structure)
        print(f"Parent's Perspective: {parent_critique_response}")
        mentor_critique_response = mentor_critique(structure)
        print(f"Mentor's Perspective: {mentor_critique_response}")
        
        # Generate improved structure
        structure_response = iterate_structure(structure, child_critique_response, parent_critique_response, mentor_critique_response)
        structure = structure_response.split("</think>")[1].strip() if "</think>" in structure_response else structure_response
        thoughts = extract_think_block(structure_response)
        
        results["iterations"].append({
            "iteration": i + 1,
            "structure": structure,
            "thoughts": thoughts,
            "child_critique": child_critique_response,
            "parent_critique": parent_critique_response,
            "mentor_critique": mentor_critique_response
        })
        
        end_time = time.time()
        results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
        print(f"=== time taken: {round((end_time - start_time)/ 60, 2)} minutes")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run iterative parenting structure development')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations to run')
    args = parser.parse_args()
    
    final_results = run_iteration_cycle(args.num_iterations)
    
    # Save results
    filename = f"creator_critic_parenting_structures_results_{args.num_iterations}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nExperiment completed! Results saved to {filename}") 