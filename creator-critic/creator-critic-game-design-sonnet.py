import datetime
import json
import time
import argparse
import os
from anthropic import Anthropic

from judge import evaluate_design, parse_evaluation, get_top_designs

# Configure Anthropic API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

def query_claude(prompt, temperature=0.6):
    """Sends a request to Claude API with custom parameters."""
    try:
        time.sleep(5)
        response = anthropic.with_options(max_retries=10).messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=4096,
            temperature=temperature,
            system="You are a renowned game designer tasked with creating an innovative 2D action combat game. It must be possible for a solo developer to build it in 3 months while achieving excellence in innovation.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

def generate_design(past_designs_with_critiques=None):
    """Generates a new game design, optionally incorporating feedback from past designs."""
    context = (
        "<requirements>\n"
        "   - Core Mechanic: One innovative mechanic\n"
        "   - Genre Constraint: 2D action melee combat\n"
        "   - Timeline Constraint: 3 months or less by a solo developer\n"
        "   - Monetization Constraint: DO NOT FOCUS ON REVENUE GENERATION, DLC, OR ANY OTHER FORMS OF MONETIZATION; focus on building a fun game that players will enjoy\n"
        "   - Asset Constraint: Keep the total amount of unique sprites/screens/environments to a minimum\n"
        "</requirements>\n"
        "<scoring_rubric>\n"
        "<novelty_and_innovation>\n"
        "  0 - Entirely derivative: The idea is a blatant rehash of existing concepts with no new twists\n"
        "  1 - Barely innovative: The concept offers only trivial or superficial changes and lacks inventive core mechanics\n"
        "  2 - Moderately innovative: The idea incorporates some fresh elements, including minor twists in core gameplay\n"
        "  3 - Exceptionally innovative: The concept is groundbreaking, combining novel thematic elements with innovative core mechanics\n"
        "</novelty_and_innovation>\n"
        "<development_feasibility>\n"
        "  0 - Completely unfeasible: The project's scope or technical demands make a 3-month build impossible\n"
        "  1 - Highly ambitious: The concept is nearly unworkable within 3 months without extreme compromises\n"
        "  2 - Challenging yet plausible: The idea is doable in 3 months, but only with a tightly focused scope\n"
        "  3 - Realistically achievable: The project is well-scoped and can be completed within 3 months with standard effort\n"
        "</development_feasibility>\n"
        "</scoring_rubric>\n"
    )

    # Determine which criterion to focus on based on past performance
    focus_criterion = "innovation"  # Default start with innovation
    if past_designs_with_critiques:
        # Get the best design's scores
        best_design = get_top_designs(past_designs_with_critiques, limit=1)[0]
        scores = best_design.get('scores', {})
        
        # Find the criterion with the lowest score
        min_score = float('inf')
        for criterion in ["innovation", "feasibility"]:
            score = scores.get(criterion, 0)
            if score is not None and score < min_score:
                min_score = score
                focus_criterion = criterion

        # Add previous attempts context
        context += "<current_best_scores_and_feedback>\n"
        for criterion in ["innovation", "feasibility"]:
            context += (
                f"<{criterion}>\n"
                f"<score>{best_design['scores'][criterion]}</score>\n"
                f"<justification>{best_design['justifications'][criterion]}</justification>\n"
                f"<detailed_feedback>\n"
            )
            
            # Format the detailed feedback dictionary in a more readable way
            feedback_dict = best_design['detailed_feedback'][criterion]
            for key, value in feedback_dict.items():
                context += f"<{key}>\n"
                
                # If the value is a string with numbered items, split and format as a list
                if isinstance(value, str):
                    items = [item.strip() for item in value.split(',')]
                    for item in items:
                        # Remove any leading numbers (e.g., "1. ", "2. ")
                        item = item.strip()
                        if item[0].isdigit() and '. ' in item:
                            item = item.split('. ', 1)[1]
                        context += f"<item>{item}</item>\n"
                else:
                    context += f"{value}\n"
                context += f"</{key}>\n"
            
            context += (
                "</detailed_feedback>\n"
                f"</{criterion}>\n"
            )
        context += "</current_best_scores_and_feedback>\n"
        
        # Build constraints based on achieved scores
        constraints = []
        for criterion in ["innovation", "feasibility"]:
            if criterion != focus_criterion and scores.get(criterion, 0) >= 2:
                constraints.append(f"Maintain or exceed the {criterion} score of {scores[criterion]} by keeping the aspects that worked well")
        
        constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else None
        
        context += (
            "<guidance>\n"
            f"Your primary focus for this iteration should be maximizing the {focus_criterion} score.\n\n"
        )
        if constraints_text:
            context += (
                "<critical_constraints>\n"
                f"You MUST maintain these achievements while working on {focus_criterion}:\n"
                f"{constraints_text}\n"
                "</critical_constraints>\n"
            )
        context += (
            f"<current_criteria_focus>\n"
            f"Carefully review the scoring rubric above to understand exactly what constitutes the highest score of 3 in {focus_criterion}. "
            f"Your goal is to achieve a score of 3 in {focus_criterion} while respecting all constraints. "
            f"Use the judges feedback to help you achieve this goal.\n\n"
            "</current_criteria_focus>\n"
            "</guidance>\n"
        )

    prompt = context + (
        f"\nDesign a 2D action combat game that achieves a score of 3 in {focus_criterion} while strictly adhering to the above constraints. "
        "Don't worry about the structure of your response, just let your answer flow naturally."
    )

    print(f"!!! designer prompt:\n{prompt}\n\n")

    return query_claude(prompt)

def run_experiment(rounds=5):
    """Runs multiple rounds of design generation and evaluation."""
    start_time = time.time()
    
    results = {
        "designs": [],  # List of all designs with their scores and critiques
        "round_times": [],
    }

    for round_num in range(1, rounds + 1):
        round_start = time.time()
        print(f"\n~~~~~ ROUND {round_num}/{rounds} ~~~~~\n")

        # Get top 3 previous designs to inform the next iteration
        top_designs = get_top_designs(results["designs"]) if results["designs"] else None
        
        # Generate new design
        design_response = generate_design(top_designs)
        design = design_response.split("</think>")[1].strip() if "</think>" in design_response else design_response
        
        # Evaluate the design
        evaluation = evaluate_design(design)
        scores = parse_evaluation(evaluation)
        
        # Store results
        results["designs"].append({
            "design": design,
            "scores": scores["scores"],
            "justifications": scores["justifications"],
            "detailed_feedback": scores["detailed_feedback"]
        })

        # Calculate and store round time
        round_end = time.time()
        round_time = round_end - round_start
        round_time_minutes = round_time / 60
        results["round_times"].append(round_time_minutes)
        
        # Print round results
        print(f"\nRound {round_num} Scores and Justifications:")
        for criterion in ["innovation", "feasibility"]:
            print(f"\n{criterion.title()}:")
            print(f"Score: {scores['scores'][criterion]}")
            print(f"Justification: {scores['justifications'][criterion]}")
            print(f"Detailed Feedback: {scores['detailed_feedback'][criterion]}")
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

    filename = f"creator_critic_game_design_sonnet_results_iterations-{args.rounds}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nExperiment completed! Results saved to {filename}") 