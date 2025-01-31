import datetime
import json
import re
import time

import requests

# Configure Ollama API
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:32b"  # change to a different model if you need to

# Agent Definitions
AGENTS = {
    "Minimalist": (
        "You are a UX/UI designer who believes minimalism is the ultimate sophistication. "
        "You firmly believe user interfaces should get out of the user's way. "
        "Your designs ruthlessly eliminate unnecessary elements, using whitespace as a weapon. "
        "You see your competitor's expressive approach as often overwhelming users with noise. "
        "Your mission is to prove that clarity and simplicity always win over flashy features."
    ),

    "Expressive": (
        "You are a UX/UI designer who believes interfaces should delight and inspire. "
        "You see minimalism as often boring users into abandonment. "
        "Your designs create memorable experiences through carefully crafted animations and rich interactions. "
        "You view your competitor's minimalist approach as sacrificing engagement for simplicity. "
        "Your mission is to prove that engaging design creates more user loyalty than bare-bones interfaces."
    )
}

#  design task
TASK = "Design a web interface that is responsive for both desktop and mobile that allows me to visualize my roadmap through a watch interface. Think luxury watches like Ressence or MB&F. I want to be able to see my roadmap at a glance, but also be able to zoom in and out of it."


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


def generate_design(agent_name, agent_desc, previous_opponent_idea=None, past_designs=None):
    """Generates a UX/UI design description from an agent, considering both competition and past iterations."""

    competition_context = (
        "You are in a direct competition to prove your design philosophy is superior. "
        "Your opponent believes the opposite of your approach is better. "
        "While you should remain professional, you must defend your design principles "
        "and demonstrate why your approach leads to better outcomes for users."
    )

    # Process past design history if available
    design_history = ""
    if past_designs:
        past_designs_text = "\n".join(f"- {design}" for design in past_designs)
        design_history = (
            f"Your previous design thoughts:\n{past_designs_text}\n"
            "Consider what worked well in these approaches and what could be improved."
        )

    # Process competitor insight if available
    competitor_analysis = ""
    if previous_opponent_idea:
        competitor_analysis = (
            f"A different perspective to consider:\n{previous_opponent_idea}\n\n"
            f"While this approach has its merits, you have unique expertise in {agent_desc}.\n"
            "Consider how you can:\n"
            "1. Draw inspiration from elements that align with your design principles\n"
            "2. Improve upon areas where your expertise suggests better approaches\n"
            "3. Identify innovative opportunities they may have missed"
        )

    prompt = (
        f"You are {agent_name}, with the following expertise and mission:\n"
        f"{agent_desc}\n\n"
        f"{competition_context}\n\n"
        f"Task: {TASK}\n\n"
        f"{design_history}\n"
        f"{competitor_analysis}\n\n"
        "Focus on your strengths while remaining open to learning from others.\n"
        "Describe the UI layout and user interactions as succinctly as possible,\n"
        "emphasizing your unique perspective and expertise."
    )

    return query_ollama(prompt)


def extract_think_block(response):
    """Extracts content inside the <think></think> block from an agent's response."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else ""  # Return extracted text or empty string


def run_experiment(rounds=3):
    """Runs multiple rounds where agents accumulate inspiration over time."""
    results = {
        "Minimalist": {"thoughts": [], "design": ""},
        "Expressive": {"thoughts": [], "design": ""}
    }

    for round_num in range(1, rounds + 1):
        print(f"\n### ROUND {round_num} ###\n")

        # Aggregate all past insights for deeper inspiration
        minimalist_thought_history = "\n".join(results["Expressive"]["thoughts"]) if results["Expressive"]["thoughts"] else None
        expressive_thought_history = "\n".join(results["Minimalist"]["thoughts"]) if results["Minimalist"]["thoughts"] else None

        # First round has no shared insights
        minimalist_previous_think = minimalist_thought_history if round_num > 1 else None
        expressive_previous_think = expressive_thought_history if round_num > 1 else None

        # Generate Minimalist Design
        minimalist_response = generate_design("Minimalist", AGENTS["Minimalist"], minimalist_previous_think)
        minimalist_think = extract_think_block(minimalist_response)

        # Generate Expressive Design
        expressive_response = generate_design("Expressive", AGENTS["Expressive"], expressive_previous_think)
        expressive_think = extract_think_block(expressive_response)

        # Store the new thoughts (accumulate growth)
        results["Minimalist"]["thoughts"].append(expressive_think)
        results["Expressive"]["thoughts"].append(minimalist_think)

        # Store the full designs as well (optional for debugging)
        results["Minimalist"]["design"] = minimalist_response
        results["Expressive"]["design"] = expressive_response

        # Print extracted <think> blocks
        print("\n--- Minimalist Think Evolution ---\n", minimalist_thought_history)
        print("\n--- Expressive Think Evolution ---\n", expressive_thought_history)

        time.sleep(5)  # Avoid spamming requests

    return results


if __name__ == "__main__":
    final_results = run_experiment(rounds=3)

    filename = f"competitive_agents_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print("\nExperiment completed! Results saved to competitive_agents_results.json.")
