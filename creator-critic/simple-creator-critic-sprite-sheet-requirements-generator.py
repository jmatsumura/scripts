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
    model = CRITIC_MODEL if is_critic else DESIGNER_MODEL
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

def generate_initial_requirements():
    """Creator agent: produce initial design document for natural language to sprite sheet system."""
    prompt = """Design Requirements for Natural Language Sprite Sheet Creator:

Core Concept: A system that recursively builds sprite sheets from natural language by solving the simplest case first.

Input: Natural language animation request (e.g., "Create a walking cat with 4 frames")
Output: Sprite sheet with N frames side by side (32x32 each)

Process (Base Case → Full Animation):
1. Single Frame Creation
   - Draw the defining pose/state of the object
   - Base case: simplest representation that captures the essence
   - Example: For walking cat, draw standing cat in profile

2. Frame Multiplication
   - Use base frame to derive variations
   - Each new frame is a small delta from previous
   - Example: Shift legs slightly for next walking position

3. Complete Sheet
   - Combine frames side by side
   - Standard format: 32x32 pixels per frame
   - Consistent spacing between frames"""

    return query_ollama(prompt)

def mathematician_critique(req):
    """Critic from a mathematical/algorithmic perspective."""
    prompt = f"""As a mathematician focused on elegant recursive solutions, review:

{req}

Suggest ONE specific improvement to make this system's core recursive logic more elegant or minimal. Be as succinct as possible, just a sentence or two please."""
    return query_ollama(prompt, is_critic=True)

def animation_theorist_critique(req):
    """Critic from an animation theory perspective."""
    prompt = f"""As an animation theorist who understands the fundamental principles of motion and keyframing, review:

{req}

Suggest ONE specific improvement to make the base case and frame transitions better capture the essence of movement. Be as succinct as possible, just a sentence or two please."""
    return query_ollama(prompt, is_critic=True)

def pixel_artist_critique(req):
    """Critic from a pixel art and sprite design perspective."""
    prompt = f"""As an experienced pixel artist who specializes in 32x32 sprite sheets, review:

{req}

Suggest ONE specific improvement to make the base frame design and transition rules better account for pixel art constraints and readability. Be as succinct as possible, just a sentence or two please."""
    return query_ollama(prompt, is_critic=True)

def prompt_engineer_critique(req):
    """Critic from a natural language interpretation perspective."""
    prompt = f"""As a prompt engineer who specializes in translating natural language into precise artistic instructions, review:

{req}

Suggest ONE specific improvement to make the natural language interpretation more precise and actionable for pixel art generation. Be as succinct as possible, just a sentence or two please."""
    return query_ollama(prompt, is_critic=True)

def compute_similarity(text1, text2):
    """Compute semantic similarity between two texts."""
    embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
    similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def iterate_requirements(prev_req, math_fb, anim_fb, pixel_fb, prompt_fb, max_tries=3):
    """Updates design requirements based on critique while maintaining similarity."""
    base_prompt = f"""Design Requirements for Natural Language Sprite Sheet Creator:

Write a complete, self-contained design document incorporating this expert feedback:
- Mathematical Elegance: {math_fb}
- Animation Fundamentals: {anim_fb}
- Pixel Art Constraints: {pixel_fb}
- Language Interpretation: {prompt_fb}

Core Concept: A system that recursively builds sprite sheets from natural language by solving the simplest case first.

Input: Natural language animation request (e.g., "Create a walking cat with 4 frames")
Output: Sprite sheet with N frames side by side (32x32 each)

Process (Base Case → Full Animation):
1. Single Frame Creation
   - Draw the defining pose/state of the object
   - Base case: simplest representation that captures the essence
   - Example: For walking cat, draw standing cat in profile

2. Frame Multiplication
   - Use base frame to derive variations
   - Each new frame is a small delta from previous
   - Example: Shift legs slightly for next walking position

3. Complete Sheet
   - Combine frames side by side
   - Standard format: 32x32 pixels per frame
   - Consistent spacing between frames"""

    for attempt in range(max_tries):
        retry_emphasis = f"\n\nIMPORTANT: Your response MUST maintain at least {SIMILARITY_THRESHOLD*100}% semantic similarity with the core concepts while being a fresh, complete design. This is attempt {attempt + 1} of {max_tries}." if attempt > 0 else ""
        
        response = query_ollama(base_prompt + retry_emphasis, temperature=0.7)
        
        # Clean response if it contains think blocks
        new_req = response.split("</think>")[1].strip() if "</think>" in response else response
        
        similarity = compute_similarity(prev_req, new_req)
        print(f"Attempt {attempt + 1}: Similarity score = {similarity:.3f}")
        
        if similarity >= SIMILARITY_THRESHOLD:
            return response
            
    return prev_req

def extract_think_block(response):
    """Extracts content inside the <think></think> block from the response."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else ""

def run_iteration_cycle(num_iterations=5):
    """Runs the iterative requirements development process."""
    results = {
        "iterations": [],
        "iteration_time": [],
        "drawing_examples": []  # Store example drawing commands and their interpretations
    }
    
    # Initial requirements
    start_time = time.time()
    req_response = generate_initial_requirements()
    requirements = req_response.split("</think>")[1].strip() if "</think>" in req_response else req_response
    thoughts = extract_think_block(req_response)
    
    end_time = time.time()
    
    results["iterations"].append({
        "iteration": 0,
        "requirements": requirements,
        "thoughts": thoughts,
        "mathematician_critique": None,
        "animation_theorist_critique": None,
        "pixel_artist_critique": None,
        "prompt_engineer_critique": None
    })
    results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
    
    # Iterative improvement cycle
    for i in range(num_iterations):
        print(f"\n=== Iteration {i+1} ===")
        start_time = time.time()
        
        # Get critiques
        mathematician_critique_response = mathematician_critique(requirements)
        print(f"Mathematician's Perspective: {mathematician_critique_response}")
        animation_theorist_critique_response = animation_theorist_critique(requirements)
        print(f"Animation Theorist's Perspective: {animation_theorist_critique_response}")
        pixel_artist_critique_response = pixel_artist_critique(requirements)
        print(f"Pixel Artist's Perspective: {pixel_artist_critique_response}")
        prompt_engineer_critique_response = prompt_engineer_critique(requirements)
        print(f"Prompt Engineer's Perspective: {prompt_engineer_critique_response}")
        
        # Generate improved requirements
        req_response = iterate_requirements(requirements, mathematician_critique_response, animation_theorist_critique_response, pixel_artist_critique_response, prompt_engineer_critique_response)
        requirements = req_response.split("</think>")[1].strip() if "</think>" in req_response else req_response
        thoughts = extract_think_block(req_response)
        
        results["iterations"].append({
            "iteration": i + 1,
            "requirements": requirements,
            "thoughts": thoughts,
            "mathematician_critique": mathematician_critique_response,
            "animation_theorist_critique": animation_theorist_critique_response,
            "pixel_artist_critique": pixel_artist_critique_response,
            "prompt_engineer_critique": prompt_engineer_critique_response
        })
        
        end_time = time.time()
        results["iteration_time"].append(round((end_time - start_time)/ 60, 2))
        print(f"=== time taken: {round((end_time - start_time)/ 60, 2)} minutes")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run iterative sprite sheet requirements development')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations to run')
    args = parser.parse_args()
    
    final_results = run_iteration_cycle(args.num_iterations)
    
    # Save results
    filename = f"creator_critic_sprite_sheet_requirements_results_{args.num_iterations}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nExperiment completed! Results saved to {filename}") 