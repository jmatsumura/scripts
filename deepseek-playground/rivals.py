import datetime
import json
import re
import time
import argparse

import requests

# Configure Ollama API
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:32b"  # change to a different model if you need to

# Agents with concise character descriptions
AGENTS = {
    "Minimalist": "You are driven by ruthless simplicity and efficient elegance. Every element must justify its existence. You believe the most powerful designs arise from reduction to pure essence.",
    "Expressive": "You are passionate about rich, immersive experiences and artistic innovation. You believe truly memorable designs require bold creative vision and carefully crafted details."
}

TASK = (
    "Design a responsive web interface (desktop & mobile) for a watch-like roadmap view inspired by luxury brands like Ressence or MB&F. "
    "The features of this app should be particularly valuable for solo gamedevs working on game jams and other ambitious projects with time constraints."
)


def generate_design(agent_name, agent_desc, competitor_past_thoughts=None, past_designs=None):
    stakes = (
        "This is a career-defining moment. The chosen design will not only become the product's signature look, "
        "but will be featured in major design publications and conferences. Your name and philosophy will become "
        "synonymous with reimagining how time-based data can be visualized. The losing design will be archived."
    )

    developer_message = (
        f"You're competing for extremely high stakes: {stakes} "
        "While you should absolutely incorporate any brilliant ideas you see, your goal is to create something "
        "so compelling that choosing any other design would be unthinkable. This is your chance to prove "
        f"that the {agent_name} philosophy is the future of interface design."
    )

    competition_context = (
        f"# Design Challenge\n{TASK}\n\n"
        f"# Your Design Philosophy\n{agent_name}: {agent_desc}\n\n"
        f"# Competition Stakes\n{developer_message}\n\n"
    )

    if past_designs:
        competition_context += "# Design Evolution\n## Your previous iterations of this design:\n"
        for i, d in enumerate(past_designs, start=0):
            if d:
                competition_context += (
                    f"--- START_OF_DESIGN_ATTEMPT_{i+1} ---\n"
                    f"{d}\n"
                    f"--- END_OF_DESIGN_ATTEMPT_{i+1} ---\n\n"
                )  

    if competitor_past_thoughts:
        competition_context += "# Latest Competition\n## Recent thoughts from your competitor's approach:\n"
        for i, t in enumerate(competitor_past_thoughts, start=0):
            if t:
                competition_context += (
                    f"--- START_OF_COMPETITOR_THOUGHTS_{i+1} ---\n"
                    f"{t}\n"
                    f"--- END_OF_COMPETITOR_THOUGHTS_{i+1} ---\n\n"
                )

    prompt = competition_context + "# Your Move\nCreate your winning design."

    print(f"Prompt: {prompt}")

    return query_ollama(prompt)


def query_ollama(prompt, temperature=0.6, top_k=50, top_p=0.95):
    """Sends a request to Ollama's API with custom parameters."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
    }

    response = requests.post(OLLAMA_ENDPOINT, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "No response")
    else:
        return f"Error: {response.status_code} - {response.text}"
    

def extract_think_block(response):
    """Extracts content inside the <think></think> block from an agent's response."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else ""  # Return extracted text or empty string


def run_experiment(rounds=3):
    """Runs multiple rounds where agents accumulate inspiration over time."""
    start_time = time.time()  # Track when experiment starts


    results = {
        "Minimalist": {"thoughts": [], "designs": [], "latest_design": ""},
        "Expressive": {"thoughts": [], "designs": [], "latest_design": ""}
    }

    for round_num in range(1, rounds + 1):
        round_start = time.time()  # Track the start of each round
        print(f"\n### ROUND {round_num} ###\n")

        # Aggregate all past insights for deeper inspiration
        # filter to just the last 5 rounds
        minimalist_thought_history = results["Expressive"]["thoughts"][-5:] if results["Expressive"]["thoughts"] else None
        expressive_thought_history = results["Minimalist"]["thoughts"][-5:] if results["Minimalist"]["thoughts"] else None

        # First round has no shared insights
        minimalist_previous_think = minimalist_thought_history if round_num > 1 else None
        expressive_previous_think = expressive_thought_history if round_num > 1 else None

        # Generate Minimalist Design, it gets thoughts from the expressive but previous designs from the minimalist
        previous_minimalist_design = results["Minimalist"]["designs"][-5:]
        minimalist_response = generate_design("Minimalist", AGENTS["Minimalist"], expressive_previous_think, previous_minimalist_design)
        minimalist_think = extract_think_block(minimalist_response)

        # Generate Expressive Design, it gets thoughts from the minimalist but previous designs from the expressive
        previous_expressive_design = results["Expressive"]["designs"][-5:]
        expressive_response = generate_design("Expressive", AGENTS["Expressive"], minimalist_previous_think, previous_expressive_design)
        expressive_think = extract_think_block(expressive_response)

        # Store the new thoughts (accumulate growth)
        results["Minimalist"]["thoughts"].append(expressive_think)
        results["Expressive"]["thoughts"].append(minimalist_think)

        # Store the full designs as well (optional for debugging)
        results["Minimalist"]["designs"].append(minimalist_response.split("</think>")[1])
        results["Expressive"]["designs"].append(expressive_response.split("</think>")[1])

        # Store the latest design
        results["Minimalist"]["latest_design"] = minimalist_response
        results["Expressive"]["latest_design"] = expressive_response

        round_end = time.time()
        print(f"Round {round_num} runtime: {(round_end - round_start) / 60} minutes")

    total_time = time.time() - start_time
    print(f"\nTotal experiment runtime: {(total_time) / 60} minutes")

    return results


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Run competitive design agents experiment')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Number of rounds to run the experiment (default: 3)')
    
    args = parser.parse_args()
    
    final_results = run_experiment(rounds=args.rounds)

    filename = f"competitive_agents_results_iterations-{args.rounds}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nExperiment completed! Results saved to {filename}")
