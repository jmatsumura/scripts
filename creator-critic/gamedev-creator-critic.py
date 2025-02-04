import datetime
import json
import re
import time
import argparse
import os

import requests

# Configure Ollama API
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")  # change to a different model if you need to

TASK = (
    "Design a 2D action combat game with one core innovative mechanic that a solo developer can build in 30 days. "
)

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

def evaluate_design(design):
    """Evaluates a game design based on the competition criteria."""
    prompt = (
        "You are an experienced game industry judge evaluating game design pitches. Your task is to critically and ruthlessly assess the concept below. Do not give the benefit of the doubt—be brutally honest in your evaluation. Score the following game design concept based on these criteria:\n\n"
        "Originality Score (0-3): How unique and fresh is the concept?\n"
        "  0 - Entirely derivative: The idea is a blatant rehash of existing concepts with no new twists.\n"
        "  1 - Barely original: The concept offers only trivial or superficial changes to familiar ideas.\n"
        "  2 - Moderately original: The idea incorporates some fresh elements but still leans on established tropes.\n"
        "  3 - Exceptionally original: The concept is innovative and groundbreaking, setting itself apart from all known designs.\n\n"
        "Development Feasibility (0-3): How realistic is it for a solo developer to build in 30 days?\n"
        "  0 - Completely unfeasible: The project's scope or technical demands make a 30-day build impossible.\n"
        "  1 - Highly ambitious: The concept is nearly unworkable within 30 days without extreme compromises.\n"
        "  2 - Challenging yet plausible: The idea is doable in 30 days, but only with a tightly focused scope.\n"
        "  3 - Realistically achievable: The project is well-scoped and can be completed within 30 days with standard effort.\n\n"
        "Core Mechanic Innovation (0-2): How novel and engaging is the core gameplay?\n"
        "  0 - Generic mechanics: The gameplay is formulaic with no inventive twist.\n"
        "  1 - Some innovation: The core mechanic shows hints of novelty but remains largely conventional.\n"
        "  2 - Highly innovative: The core gameplay is fresh and redefines what players can expect.\n\n"
        f"Game Design:\n{design}\n\n"
        "Respond with your scores and a brief justification in this exact JSON format and nothing more:\n"
        "{\n"
        "  \"scores\": {\n"
        "    \"originality\": [score],\n"
        "    \"feasibility\": [score],\n"
        "    \"innovation\": [score]\n"
        "  },\n"
        "  \"justification\": \"[a few sentences explaining the scoring]\"\n"
        "}\n"
    )
    
    return query_ollama(prompt)

def parse_evaluation(evaluation):
    """Extracts scores and justification from evaluation response."""
    # First try to extract content between ```json and ``` markers
    json_match = re.search(r'```json\s*({[\s\S]*?})\s*```', evaluation)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            return result
        except json.JSONDecodeError:
            pass
    
    try:
        # Try to parse the entire response as JSON
        return json.loads(evaluation)
    except json.JSONDecodeError:
        print("Warning: Could not parse evaluation as JSON")
        return {
            "scores": {},
            "justification": evaluation
        }

def extract_think_block(response):
    """Extracts content inside the <think></think> block from the response."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else ""

def generate_design(task, past_designs_with_critiques=None):
    """Generates a new game design, optionally incorporating feedback from past designs."""
    context = (
        "<role>\n"
        "You are a world-class game designer known for creating innovative yet feasible games. "
        "Your expertise lies in finding the perfect balance between creativity and practicality.\n"
        "</role>\n\n"
        "<challenge>\n"
        f"{task}\n"
        "</challenge>\n\n"
        "<scoring_rubric>\n"
        "Originality Score (0-3):\n"
        "  0 - Entirely derivative: The idea is a blatant rehash of existing concepts with no new twists\n"
        "  1 - Barely original: The concept offers only trivial or superficial changes to familiar ideas\n"
        "  2 - Moderately original: The idea incorporates some fresh elements but still leans on established tropes\n"
        "  3 - Exceptionally original: The concept is innovative and groundbreaking, setting itself apart from all known designs\n\n"
        "Development Feasibility (0-3):\n"
        "  0 - Completely unfeasible: The project's scope or technical demands make a 30-day build impossible\n"
        "  1 - Highly ambitious: The concept is nearly unworkable within 30 days without extreme compromises\n"
        "  2 - Challenging yet plausible: The idea is doable in 30 days, but only with a tightly focused scope\n"
        "  3 - Realistically achievable: The project is well-scoped and can be completed within 30 days with standard effort\n\n"
        "Core Mechanic Innovation (0-2):\n"
        "  0 - Generic mechanics: The gameplay is formulaic with no inventive twist\n"
        "  1 - Some innovation: The core mechanic shows hints of novelty but remains largely conventional\n"
        "  2 - Highly innovative: The core gameplay is fresh and redefines what players can expect\n"
        "</scoring_rubric>\n\n"
    )

    # Determine which criterion to focus on based on past performance
    focus_criterion = "originality"  # Default start with originality
    if past_designs_with_critiques:
        # Get the best scores achieved for each criterion
        best_scores = {
            "originality": 0,
            "feasibility": 0,
            "innovation": 0
        }
        for entry in past_designs_with_critiques:
            scores = entry.get('scores', {})
            for criterion, score in scores.items():
                best_scores[criterion] = max(best_scores[criterion], score)
        
        # Progressive focus based on achievement
        if best_scores["originality"] < 3:
            focus_criterion = "originality"
        elif best_scores["feasibility"] < 3:
            focus_criterion = "feasibility"
        else:
            focus_criterion = "innovation"

        # Add previous attempts context
        top_designs = get_top_designs(past_designs_with_critiques, limit=2)
        context += "<previous_attempts>\n"
        for i, entry in enumerate(top_designs, 1):
            context += (
                f"<attempt_number>{i}</attempt_number>\n"
                f"<design>{entry['design']}</design>\n"
                f"<scores>{json.dumps(entry['scores'], indent=2)}</scores>\n"
                f"<justification>{entry['justification']}</justification>\n\n"
            )
        
        context += "</previous_attempts>\n\n"
        
        # Build constraints based on achieved scores
        constraints = []
        if best_scores["originality"] >= 3 and focus_criterion != "originality":
            constraints.append("Keep the originality score at 3 by retaining the unique aspects that made previous designs stand out")
        if best_scores["feasibility"] >= 3 and focus_criterion != "feasibility":
            constraints.append("Maintain the feasibility score at 3 by keeping the scope realistic for a 30-day project")
        if best_scores["innovation"] >= 2 and focus_criterion != "innovation":
            constraints.append("Preserve the innovation score at 2 by keeping the core mechanic fresh and engaging")
        
        constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else "No constraints yet - focus on achieving high scores"
        
        context += (
            "<guidance>\n"
            f"Your primary focus for this iteration should be maximizing the {focus_criterion} score. "
            f"Previous attempts have achieved these best scores: {json.dumps(best_scores, indent=2)}.\n\n"
            f"CRITICAL CONSTRAINTS - You MUST maintain these achievements while working on {focus_criterion}:\n"
            f"{constraints_text}\n\n"
            f"Your goal is to achieve a perfect score in {focus_criterion} while preserving the above constraints. "
            f"Carefully review the scoring rubric above to understand exactly what constitutes a perfect score in {focus_criterion}. "
            "Study the previous attempts and their critiques carefully to understand what worked and what didn't.\n"
            "</guidance>\n\n"
        )

    prompt = context + (
        "<task>\n"
        f"Create a new game design that achieves a perfect score in {focus_criterion} while strictly adhering to the above constraints. "
        "Before presenting your design, verify it maintains high scores already achieved in previous iterations by checking against the scoring rubric.\n\n"
        "Now, present your design concept.\n"
        "</task>"
    )

    print(prompt)

    return query_ollama(prompt)

def get_top_designs(designs_with_scores, limit=3):
    """Returns the top N designs based on total score."""
    def calculate_total_score(entry):
        scores = entry.get('scores', {})
        return sum(scores.values())
    
    sorted_designs = sorted(designs_with_scores, key=calculate_total_score, reverse=True)
    return sorted_designs[:limit]

def run_experiment(rounds=5, task=TASK):
    """Runs multiple rounds of design generation and evaluation."""
    start_time = time.time()
    
    results = {
        "designs": [],  # List of all designs with their scores and critiques
        "round_times": [],
        "task": task
    }

    for round_num in range(1, rounds + 1):
        round_start = time.time()
        print(f"\n~~~~~ ROUND {round_num}/{rounds} ~~~~~\n")

        # Get top 3 previous designs to inform the next iteration
        top_designs = get_top_designs(results["designs"]) if results["designs"] else None
        
        # Generate new design
        design_response = generate_design(task, top_designs)
        design = design_response.split("</think>")[1].strip() if "</think>" in design_response else design_response
        thoughts = extract_think_block(design_response)
        
        # Evaluate the design
        evaluation = evaluate_design(design)
        scores = parse_evaluation(evaluation)
        
        # Store results
        results["designs"].append({
            "design": design,
            "thoughts": thoughts,
            "scores": scores["scores"],
            "justification": scores["justification"]
        })

        # Calculate and store round time
        round_end = time.time()
        round_time = round_end - round_start
        round_time_minutes = round_time / 60
        results["round_times"].append(round_time_minutes)
        
        # Print round results
        print(f"\nRound {round_num} Scores:")
        print(json.dumps(scores["scores"], indent=2))
        print(f"\nRound {round_num} runtime: {round_time_minutes:.2f} minutes")

    total_time = time.time() - start_time
    print(f"\nTotal experiment runtime: {(total_time) / 60:.2f} minutes")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run creator-critic game design experiment')
    parser.add_argument('--rounds', type=int, default=5,
                       help='Number of rounds to run the experiment (default: 5)')
    
    args = parser.parse_args()
    
    final_results = run_experiment(rounds=args.rounds)

    filename = f"gamedev_creator_critic_results_iterations-{args.rounds}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nExperiment completed! Results saved to {filename}") 