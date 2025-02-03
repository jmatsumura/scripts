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
    "Systematic": (
        "you believe in precision and structure. Your design philosophy centers on a single, well-tuned gameplay loop. "
        "Every element is methodically balanced to deliver a polished, reliable experience that can be rapidly developed."
    ),
    "Whimsical": (
        "you thrive on creative chaos and unexpected twists. Your game concepts turn even a simple mechanic into a playground "
        "of quirky surprises and bold, imaginative ideas that defy conventional design norms."
    )
}

TASK_BROAD = (
    "Design a 2D game that a solo developer can create in under 30 days. "
    "Focus on one core gameplay mechanic and minimal screens in order to keep scope as small as possible."
)

TASK_WITH_GENRE_DEFINED = (
    "Design a 2D action combat game that a solo developer can create in under 30 days. "
    "Focus on one core gameplay mechanic and minimal screens in order to keep scope as small as possible."
)

TASK_WITH_GENRE_AND_LOOT_DEFINED = (
    "Design a 2D action combat game with loot progression that a solo developer can create in under 30 days. "
    "Focus on one core gameplay mechanic and minimal screens in order to keep scope as small as possible."
)

def generate_design(agent_name, agent_desc, task, competitor_past_thoughts=None, past_designs=None):
    stakes = (
        "this is a make-or-break moment. Your entire game studio is hanging by a thread. "
        "Failure means shuttering your studio."
    )

    developer_message = (
        f"You are locked in a game design pitch competition with another game studio to secure funding: {stakes} "
        "While you may draw inspiration from your rival’s ideas, your mission is to prove that the "
        f"{agent_name} design philosophy produces a better game. Leave nothing to chance and do whatever it takes to outshine your competitor."
    )

    competition_context = (
        f"# Game Design Challenge\n{task}\n\n"
        f"# Competition Stakes\n{developer_message}\n\n"
        f"# Evaluation & Judging Framework\nWinner is determined by the highest total score from the following criteria:\n- Originality Score (0-3)\n- Development Feasibility (0-3)\n- Core Mechanic Innovation (0-2)\n- Market Potential (0-2)\n\n"
    )

    if past_designs:
        competition_context += "# Game Concept Evolution\n## Your previous designs:\n"
        for i, d in enumerate(past_designs, start=1):
            if d:
                competition_context += (
                    f"--- START_OF_DESIGN_ATTEMPT_{i} ---\n"
                    f"{d}\n"
                    f"--- END_OF_DESIGN_ATTEMPT_{i} ---\n\n"
                )

    if competitor_past_thoughts:
        competition_context += "# Competitor Insights\n## Thoughts from your competition:\n"
        for i, t in enumerate(competitor_past_thoughts, start=1):
            if t:
                competition_context += (
                    f"--- START_OF_COMPETITOR_THOUGHTS_{i} ---\n"
                    f"{t}\n"
                    f"--- END_OF_COMPETITOR_THOUGHTS_{i} ---\n\n"
                )

    prompt = competition_context + (
        f"# Your Move\nAct as a {agent_name} game designer, {agent_desc}\n\nPresent your winning game design concept as briefly as possible."
    )

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


def run_experiment(rounds=3, task=TASK_BROAD, include_past_designs=False):
    """Runs multiple rounds where agents accumulate inspiration over time."""
    start_time = time.time()  # Track when experiment starts


    results = {
        "Systematic": {"competitor_thoughts": [], "designs": [], "latest_design": ""},
        "Whimsical": {"competitor_thoughts": [], "designs": [], "latest_design": ""},
        "round_times": [],
        "task": task
    }

    for round_num in range(1, rounds + 1):
        round_start = time.time()  # Track the start of each round
        print(f"\n### ROUND {round_num} ###\n")

        # Aggregate all past insights for deeper inspiration
        # filter to just the last 5 rounds
        agent_1_competitor_thought_history = results["Whimsical"]["competitor_thoughts"][-5:] if results["Whimsical"]["competitor_thoughts"] else None
        agent_2_competitor_thought_history = results["Systematic"]["competitor_thoughts"][-5:] if results["Systematic"]["competitor_thoughts"] else None

        # First round has no shared insights
        agent_1_previous_competitor_thought_history = agent_1_competitor_thought_history if round_num > 1 else None
        agent_2_previous_competitor_thought_history = agent_2_competitor_thought_history if round_num > 1 else None

        # Generate Minimalist Design, it gets thoughts from the expressive but previous designs from the minimalist
        previ_1ous_agent_design = results["Systematic"]["designs"][-5:] if include_past_designs else None
        agent_1_response = generate_design("Systematic", AGENTS["Systematic"], task, agent_2_previous_competitor_thought_history, previ_1ous_agent_design)
        agent_1_think = extract_think_block(agent_1_response)

        # Generate Expressive Design, it gets thoughts from the minimalist but previous designs from the expressive
        agent_2_design = results["Whimsical"]["designs"][-5:] if include_past_designs else None
        agent_2_response = generate_design("Whimsical", AGENTS["Whimsical"], task, agent_1_previous_competitor_thought_history, agent_2_design)
        agent_2_think = extract_think_block(agent_2_response)

        # Store the new thoughts (accumulate growth)
        results["Systematic"]["competitor_thoughts"].append(agent_2_think)
        results["Whimsical"]["competitor_thoughts"].append(agent_1_think)

        # Store the design history as well
        results["Systematic"]["designs"].append(agent_1_response.split("</think>")[1])
        results["Whimsical"]["designs"].append(agent_2_response.split("</think>")[1])

        # Store the latest thought and design
        results["Systematic"]["latest_design"] = agent_1_response
        results["Whimsical"]["latest_design"] = agent_2_response

        round_end = time.time()
        round_time = round_end - round_start
        round_time_minutes = round_time / 60
        results["round_times"].append(round_time_minutes)
        print(f"Round {round_num} runtime: {round_time_minutes} minutes")

    total_time = time.time() - start_time
    print(f"\nTotal experiment runtime: {(total_time) / 60} minutes")

    return results


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Run competitive design agents experiment')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Number of rounds to run the experiment (default: 3)')
    parser.add_argument('--task', type=str, default="broad", choices=["broad", "genre", "genre-loot"],
                       help='Task to run the experiment (default: broad)')
    parser.add_argument('--include-past-designs', type=bool, default=False,
                       help='Include past designs in the experiment (default: False)')
    
    args = parser.parse_args()
    
    final_results = run_experiment(rounds=args.rounds, task=TASK_BROAD if args.task == "broad" else TASK_WITH_GENRE_DEFINED if args.task == "genre" else TASK_WITH_GENRE_AND_LOOT_DEFINED, include_past_designs=args.include_past_designs)

    filename = f"gamedev_competitive_agents_results_iterations-{args.rounds}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nExperiment completed! Results saved to {filename}")
