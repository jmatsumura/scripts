import json
import requests
import argparse
import time
import re
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Configure Ollama API
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
CREATOR_MODEL = "hf.co/bartowski/cognitivecomputations_Dolphin3.0-R1-Mistral-24B-GGUF:Q6_K_L"
CRITIC_MODEL = "mistral-nemo"
SIMILARITY_THRESHOLD = 0.8

# Initialize the sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def query_ollama(prompt, temperature=0.6, is_critic=False):
    """Sends a request to Ollama's API."""
    if is_critic:
        # Simple prompt for critic
        formatted_prompt = prompt
        model = CRITIC_MODEL
    else:
        # Dolphin format for creator
        formatted_prompt = f"""<|im_start|>system
You are Dolphin, an AI assistant that helps game designers, trained to specialize in reasoning and first-principles analysis.

When responding, always format your replies using <think>{'{'}reasoning{'}'}</think>{'{'}answer{'}'}.
Use at least 6 reasoning steps and perform a root cause analysis before answering.
However, if the answer is very easy and requires little thought, you may leave the <think></think> block empty.

Your responses should be detailed and structured with rich Markdown formatting.
Be extensive in your explanations, just as the greatest scientific minds would be.
Always reason through the problem first, unless it's trivial, in which case you may answer directly.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        model = CREATOR_MODEL

    payload = {
        "model": model,
        "prompt": formatted_prompt,
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
    """Generates the initial video game design."""
    prompt = """Act as a highly creative yet practical video game designer, focused on producing original and engaging 2D video game concepts.

# Core Rules
- Do **not** use time loops, rewinds, echoes, or recursion-based mechanics.
- Keep the world and mechanics **forward-moving** rather than cyclical.
- Every design must include **one new video gameplay element** not commonly seen in 2D video games.

# Your task
Generate a unique **2D video game concept** with a clear and practical structure:
- **Core Concept:** The central idea or hook. Avoid themes of time manipulation.
- **Main Mechanic:** The defining video gameplay element that makes it unique.
- **Basic video Gameplay Progression:** How the video game evolves as players move forward.
- **Art & Aesthetic:** Describe the video game's look and feel, avoiding nostalgic "retro" clichés unless reimagined.
- **Development Scope:** What makes this idea feasible yet innovative for a small team?

Think expansively—this is the first concept draft, so explore creative, unexpected directions!"""

    return query_ollama(prompt)

def refinement_critique(design):
    """Reviews the video game design and suggests one way to make it more polished."""
    prompt = f"""Act as a video game design critic focused on feasibility and balance.
    
Evaluate the following video game concept and suggest improvements to make it more structured and enjoyable to play.

Game Concept:
{design}

Provide specific feedback in just a sentence or two on the following:
- One gameplay mechanic that is unclear, difficult to implement, or simply not fun for the player."""

    return query_ollama(prompt, is_critic=True)

def divergent_critique(design):
    """Reviews the video game design and suggests one way to make it more fun."""
    prompt = f"""Act as a video game design critic focused on originality.
    
Evaluate the following video game concept and suggest ways to make it more unexpected and creatively fresh.

Game Concept:
{design}

Provide specific feedback in just a sentence or two on the following:
- Suggest one entirely new video gameplay twist that fits the concept."""

    return query_ollama(prompt, is_critic=True)

def compute_similarity(text1, text2):
    """Compute semantic similarity between two texts."""
    # Get embeddings
    embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def iterate_design(previous_design, divergent_feedback, refinement_feedback, max_attempts=3):
    """Updates the video game design based on the critique with similarity checking."""
    base_prompt = f"""Act as a highly creative yet practical video game designer, refining and evolving an existing 2D video game concept.

# Core Rules
- Do **not** use time loops, rewinds, echoes, or recursion-based mechanics.
- Every design must include **one new video gameplay element** not commonly seen in 2D video games.
- Keep the world and mechanics **forward-moving** rather than cyclical.

# Your Task
Refine the following **2D video game concept**, keeping its strengths while meaningfully improving it based on expert feedback.

## Previous Concept:
{previous_design}

## Critic Feedback:
- **Originality Critic Suggests:** {divergent_feedback}
- **Refinement Critic Suggests:** {refinement_feedback}

## Revised video Game Concept:
Rework and expand on the previous concept while keeping it structurally similar. Your new version should include:
- **Core Concept:** Maintain the essence of the video game while introducing small, fresh elements that add originality.
- **Main Mechanic:** Adjust or refine the defining video gameplay element to improve engagement.
- **Basic video Gameplay Progression:** Ensure video gameplay remains rewarding, evolving meaningfully over time.
- **Art & Aesthetic:** Describe the look and feel of the video game, ensuring thematic cohesion.
- **Development Scope:** Consider feasibility for a small team while preserving innovation.

Each iteration must **refine rather than reinvent** the video game concept. Maintain the same structure and core ideas while enhancing clarity, depth, and uniqueness."""

    for attempt in range(max_attempts):
        # Add increasing emphasis on similarity for each retry
        retry_emphasis = f"\n\nIMPORTANT: Your response MUST maintain at least {SIMILARITY_THRESHOLD*100}% semantic similarity with the previous concept. This is attempt {attempt + 1} of {max_attempts}." if attempt > 0 else ""
        
        response = query_ollama(base_prompt + retry_emphasis)
        
        # Clean response if it contains think blocks
        new_design = response.split("</think>")[1].strip() if "</think>" in response else response
        
        # Check similarity
        similarity = compute_similarity(previous_design, new_design)
        print(f"Attempt {attempt + 1}: Similarity score = {similarity:.3f}")
        
        if similarity >= SIMILARITY_THRESHOLD:
            return response
            
    # If we've exhausted all attempts, return the previous design and let it try again
    print(f"Warning: Could not achieve desired similarity after {max_attempts} attempts.")
    return previous_design

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
        "refinement_critique": None,
        "divergent_critique": None,
    })
    results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
    
    # Iterative improvement cycle
    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} ===")
        start_time = time.time()
        
        # Get critique
        refinement_critique_response = refinement_critique(design)
        print(f"Refinement Critique: {refinement_critique_response}")
        divergent_critique_response = divergent_critique(design)
        print(f"Divergent Critique: {divergent_critique_response}")
        
        # Generate improved design
        design_response = iterate_design(design, divergent_critique_response, refinement_critique_response)
        design = design_response.split("</think>")[1].strip() if "</think>" in design_response else design_response
        thoughts = extract_think_block(design_response)
        
        results["iterations"].append({
            "iteration": i + 1,
            "design": design,
            "thoughts": thoughts,
            "refinement_critique": refinement_critique_response,
            "divergent_critique": divergent_critique_response,
        })
        
        end_time = time.time()
        results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
        print(f"=== time taken: {round((end_time - start_time)/ 60, 2)} minutes")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run iterative video game design experiment')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations to run')
    args = parser.parse_args()
    
    final_results = run_iteration_cycle(args.num_iterations)
    
    # Save results
    filename = f"creator_critic_mistral_results_{args.num_iterations}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nExperiment completed! Results saved to {filename}") 