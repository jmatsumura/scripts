import datetime
import json
import re
import time
import argparse
import os
import csv
from typing import List, Dict

import requests

# Configure Ollama API
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")  # change to a different model if you need to

# Agents with concise character descriptions
AGENTS = {
    "Systematic": (
        "you believe in precision and structure. Your design philosophy centers on a single, well-tuned gameplay loop. "
    ),
    "Whimsical": (
        "you thrive on creative chaos and unexpected twists. Your design philosophy is to turn even a simple mechanic into playgrounds which defy conventional design norms."
    )
}


TASK = (
    "Design a 2D action combat game that a solo developer can build in 30 days. "
)

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
    
    # Fallback to the original parsing logic if JSON extraction fails
    scores = {}
    lines = evaluation.split('\n')
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            try:
                if key in ['originality', 'feasibility', 'innovation']:
                    scores[key] = float(value)
            except ValueError:
                continue
    
    # Extract the justification using regex
    justification = ""
    if "JUSTIFICATION:" in evaluation:
        justification = evaluation.split("JUSTIFICATION:")[1].strip()
    
    return {
        "scores": scores or {},
        "justification": justification or ""
    }

def generate_design(agent_name, agent_desc, task, competitor_insights=None):
    developer_message = (
        "You are in a game design pitch competition with another studio to secure funding. "
        "This is a make-or-break moment. "
        "Losing mean shuttering your studio and the end of your lifelong dream of making games."
        "Leave nothing to chance and do whatever it takes to beat your competitor."
    )

    competition_context = (
        f"Act as a world-renowned game designer who prefers a {agent_name.lower()} approach to design: {agent_desc}\n\n"
        f"<game_design_challenge>\n{task}\n</game_design_challenge>\n\n"
        f"<evaluation_judging_framework>\nThe winner is determined by the highest total score aggregated across the following criteria:\n- Originality Score (0-3)\n- Development Feasibility (0-3)\n- Core Mechanic Innovation (0-2)\n</evaluation_judging_framework>\n\n"
    )

    if competitor_insights and len(competitor_insights) > 0:
        insight = competitor_insights[-1]  # Get the latest insight
        competition_context += (
            f"<competitor_insight>\n"
            "You can see how well your competitor has done so far as well as their thoughts that led to those scores. "
            "Use this information to your advantage to derive inspiration from their thoughts if they're scoring well. "
            "If they're not scoring well, use this information to know what to avoid."
            f"<their_best_scores_achieved>\n{insight['scores']}\n</their_best_scores_achieved>\n\n"
            f"<judgement_behind_their_scores>\n{insight['justification']}\n</judgement_behind_their_scores>\n\n"
            f"<thoughts_that_led_to_those_scores>\n{insight['thoughts']}\n</thoughts_that_led_to_those_scores>\n\n"
            f"</competitor_insight>\n\n"
        )

    prompt = competition_context + (
        f"<competition_stakes>\n{developer_message}\n</competition_stakes>\n\n"
        "Come up with a game design that will beat your competitor and keep your dream alive."
    )

    print(prompt)

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
    return match.group(1).strip() if match else ""

def run_experiment(rounds=3, task=TASK):
    """Runs multiple rounds where agents accumulate inspiration over time."""
    start_time = time.time()

    results = {
        "Systematic": {
            "competitor_insights": [],  # List of dicts with thoughts and scores
            "designs": [],  # List of full designs
            "scores": [],  # List of evaluation results
            "latest_design": ""
        },
        "Whimsical": {
            "competitor_insights": [],
            "designs": [],
            "scores": [],
            "latest_design": ""
        },
        "round_times": [],
        "task": task
    }

    for round_num in range(1, rounds + 1):
        round_start = time.time()
        print(f"\n~~~~~ ROUND {round_num}\n")

        # Get the top-scoring insights from previous rounds
        def get_top_insights(insights, scores, limit=2):
            if not insights or not scores:
                return None
            
            # Pair insights with their scores and calculate a composite score
            def calculate_composite_score(score_dict):
                if not score_dict or 'scores' not in score_dict:
                    return 0
                scores = score_dict['scores']
                return sum(scores.values())  # Sum of all individual scores
            
            # Pair insights with their composite scores and sort
            paired = [(i, s) for i, s in zip(insights, scores)]
            sorted_pairs = sorted(paired, key=lambda x: calculate_composite_score(x[1]), reverse=True)
            
            return [{"thoughts": p[0]["thoughts"], "scores": p[1]["scores"], "justification": p[1]["justification"]} for p in sorted_pairs[:limit]]

        # Generate designs with access to competitor's best thoughts
        agent_1_response = generate_design(
            "Systematic", 
            AGENTS["Systematic"], 
            task, 
            get_top_insights(
                results["Whimsical"]["competitor_insights"],
                results["Whimsical"]["scores"]
            )
        )
        
        # Evaluate Systematic design
        agent_1_design = agent_1_response.split("</think>")[1].strip()
        agent_1_evaluation = evaluate_design(agent_1_design)
        agent_1_scores = parse_evaluation(agent_1_evaluation)
        
        # Generate Whimsical design
        agent_2_response = generate_design(
            "Whimsical", 
            AGENTS["Whimsical"], 
            task,
            get_top_insights(
                results["Systematic"]["competitor_insights"],
                results["Systematic"]["scores"]
            )
        )
        
        # Evaluate Whimsical design
        agent_2_design = agent_2_response.split("</think>")[1].strip()
        agent_2_evaluation = evaluate_design(agent_2_design)
        agent_2_scores = parse_evaluation(agent_2_evaluation)

        # Store results
        agent_1_thoughts = extract_think_block(agent_1_response)
        agent_2_thoughts = extract_think_block(agent_2_response)

        # Store the thoughts in competitor insights (cross-pollinate) with full evaluation details
        results["Whimsical"]["competitor_insights"].append({
            "thoughts": agent_1_thoughts,
            "scores": agent_1_scores["scores"],
            "justification": agent_1_scores["justification"]
        })  # Whimsical gets Systematic's thoughts and scores
        results["Systematic"]["competitor_insights"].append({
            "thoughts": agent_2_thoughts,
            "scores": agent_2_scores["scores"],
            "justification": agent_2_scores["justification"]
        })  # Systematic gets Whimsical's thoughts and scores

        # Store designs and scores
        results["Systematic"]["designs"].append(agent_1_design)
        results["Systematic"]["scores"].append(agent_1_scores)
        results["Systematic"]["latest_design"] = agent_1_design

        results["Whimsical"]["designs"].append(agent_2_design)
        results["Whimsical"]["scores"].append(agent_2_scores)
        results["Whimsical"]["latest_design"] = agent_2_design

        round_end = time.time()
        round_time = round_end - round_start
        round_time_minutes = round_time / 60
        results["round_times"].append(round_time_minutes)
        print(f"~~~~~ Round {round_num} runtime: {round_time_minutes:.2f} minutes")

        # Print current round scores
        print(f"\n~~~~~ Round {round_num} Scores:")
        print(f"~~~~~ Systematic: {agent_1_scores['scores']}")
        print(f"~~~~~ Whimsical: {agent_2_scores['scores']}")

    total_time = time.time() - start_time
    print(f"\nTotal experiment runtime: {(total_time) / 60:.2f} minutes")

    return results

def json_to_csv(input_files: List[str], labels: List[str], output_file: str = None) -> None:
    """Convert game design experiment JSON results to CSV format.
    
    Args:
        input_files: List of JSON file paths to process
        labels: List of labels corresponding to each JSON file
        output_file: Optional output CSV file path. If not provided, will generate based on timestamp.
    """
    if len(input_files) != len(labels):
        raise ValueError("Number of input files must match number of labels")
        
    if not output_file:
        output_file = f"gamedev_scores_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    csv_rows = []
    
    for json_file, label in zip(input_files, labels):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        for agent in ["Systematic", "Whimsical"]:
            for iteration, score_data in enumerate(data[agent]["scores"], 1):
                if "scores" in score_data:
                    scores = score_data["scores"]
                    csv_rows.append({
                        "originality_score": scores.get("originality", ""),
                        "feasibility_score": scores.get("feasibility", ""),
                        "innovation_score": scores.get("innovation", ""),
                        "design_iteration": iteration,
                        "total_score": sum(scores.values()),
                        "agent": agent,
                        "label": label
                    })
    
    if csv_rows:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["originality_score", "feasibility_score", 
                                                 "innovation_score", "design_iteration", 
                                                 "total_score", "agent", "label"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Scores extracted to {output_file}")
    else:
        print("No scores found to extract")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run competitive design agents experiment or process results')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Number of rounds to run the experiment (default: 3)')
    parser.add_argument('--to-csv', action='store_true',
                       help='Convert JSON results to CSV instead of running experiment')
    parser.add_argument('--input-files', nargs='+',
                       help='JSON files to process when using --to-csv')
    parser.add_argument('--labels', nargs='+',
                       help='Labels for each input file when using --to-csv')
    parser.add_argument('--output-file',
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    if args.to_csv:
        if not args.input_files or not args.labels:
            parser.error("--to-csv requires --input-files and --labels")
        json_to_csv(args.input_files, args.labels, args.output_file)
    else:
        final_results = run_experiment(rounds=args.rounds)
        filename = f"gamedev_competitive_agents_with_judge_results_iterations-{args.rounds}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
        
        print(f"\nExperiment completed! Results saved to {filename}") 