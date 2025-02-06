import json
import os
import time
from anthropic import Anthropic

# Configure Anthropic API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

def query_claude(prompt, system_prompt, temperature=0):
    """Sends a request to Claude API with custom parameters."""
    try:
        response = anthropic.with_options(max_retries=10).messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_design(design):
    """Evaluates a game design based on the competition criteria."""
    
    def evaluate_innovation(design):
        system_prompt = (
            "# Game Judge: Innovation Assessment\n\n"
            "Act as an experienced game industry judge evaluating game design pitches for INNOVATION ONLY.\n\n"
            "## Scoring Rubric (using the standard 0-3 scale)\n"
            "  0 - Entirely derivative: The idea is a blatant rehash of existing concepts with no new twists\n"
            "  1 - Barely innovative: The concept offers only trivial or superficial changes and lacks inventive core mechanics\n"
            "  2 - Moderately innovative: The idea incorporates some fresh elements, including minor twists in core gameplay\n"
            "  3 - Exceptionally innovative: The concept is groundbreaking, combining novel thematic elements with innovative core mechanics\n\n"
            "## Task\n"
            "Ruthlessly assess how innovative the core gameplay mechanic is based on the above rubric.\n"
            "Remember, the 0-3 scale is commonly used in formal assessments—choose exactly one of the specified score values, no other values are allowed.\n"
            "The designer you are judging is one of the best there ever was, so do not be too generous.\n\n"
            "## Feedback Guidelines\n"
            "1. Identify specific innovative elements and explain why they are or aren't groundbreaking\n"
            "2. Compare with existing games to highlight similarities or unique differences\n"
            "3. Suggest concrete ways to enhance innovation while maintaining the core concept\n"
            "4. Consider both mechanical innovation and thematic innovation separately\n"
            "5. Evaluate how well the innovative elements serve the game's core experience"
        )
        
        prompt = (
            "<design_to_evaluate>\n"
            f"{design}\n\n"
            "</design_to_evaluate>\n\n"
            "Please evaluate and respond with your assessment in this exact JSON format and nothing more:\n"
            "{\n"
            "  \"score\": [MUST be exactly 0, 1, 2, or 3 as an integer number, not a string],\n"
            "  \"justification\": \"[2-3 sentences explaining the overall score]\",\n"
            "  \"innovative_elements\": \"[List and analyze specific innovative mechanics/features]\",\n"
            "  \"comparison\": \"[Compare with existing games, highlighting unique differences]\",\n"
            "  \"improvement_suggestions\": \"[Concrete suggestions to enhance innovation]\"\n"
            "}\n"
        )
        print(f"!!! innovation system prompt:\n{system_prompt}\n\n")
        print(f"!!! innovation prompt:\n{prompt}\n\n")
        return query_claude(prompt, system_prompt)

    def evaluate_feasibility(design):
        system_prompt = (
            "# Game Judge: Feasibility Assessment\n\n"
            "Act as an experienced game industry judge evaluating game design pitches for FEASIBILITY ONLY.\n\n"
            "## Scoring Rubric\n"
            "* 0 - Completely unfeasible: The project's scope or technical demands make a 3-month build impossible\n"
            "* 1 - Highly ambitious: The concept is nearly unworkable within 3 months without extreme compromises\n"
            "* 2 - Challenging yet plausible: The idea is doable in 3 months, but only with a tightly focused scope\n"
            "* 3 - Realistically achievable: The project is well-scoped and can be completed within 3 months with standard effort\n\n"
            "## Task\n"
            "Ruthlessly assess how feasible the concept is for a solo developer to build in 3 months based on the above rubric.\n"
            "The designer you are judging is one of the best there ever was, do not be too generous\n"
            "You must choose exactly one of the specified score values - no other values are allowed.\n\n"
            "## Feedback Guidelines\n"
            "1. Break down the implementation into major technical components\n"
            "2. Identify specific technical challenges and their complexity\n"
            "3. Suggest concrete scope adjustments to improve feasibility\n"
            "4. Consider asset creation time (sprites, animations, etc.)\n"
            "5. Evaluate core systems that would need to be built from scratch"
        )
        
        prompt = (
            "<design_to_evaluate>\n"
            f"{design}\n\n"
            "</design_to_evaluate>\n\n"
            "Please evaluate and respond with your assessment in this exact JSON format and nothing more:\n"
            "{\n"
            "  \"score\": [MUST be exactly 0, 1, 2, or 3 as an integer number, not a string],\n"
            "  \"justification\": \"[2-3 sentences explaining the overall score]\",\n"
            "  \"technical_breakdown\": \"[List major technical components and their complexity]\",\n"
            "  \"critical_challenges\": \"[Identify specific technical challenges]\",\n"
            "  \"scope_suggestions\": \"[Concrete suggestions to improve feasibility]\"\n"
            "}\n"
        )
        print(f"!!! feasibility system prompt:\n{system_prompt}\n\n")
        print(f"!!! feasibility prompt:\n{prompt}\n\n")
        return query_claude(prompt, system_prompt)

    # Run each evaluation separately
    innovation_result = evaluate_innovation(design)
    time.sleep(5)
    feasibility_result = evaluate_feasibility(design)

    print(f"judge results: {feasibility_result}, {innovation_result}")

    # Combine the results
    return {
        "feasibility": feasibility_result,
        "innovation": innovation_result
    }

def parse_evaluation(evaluation):
    """Extracts scores and feedback from evaluation response."""
    def parse_single_result(result_str):
        try:
            result = json.loads(result_str)
            return {
                "score": result.get("score"),
                "justification": result.get("justification", ""),
                "detailed_feedback": {
                    k: v for k, v in result.items() 
                    if k not in ["score", "justification"]
                }
            }
        except json.JSONDecodeError:
            print("Warning: Could not parse evaluation as JSON")
            return {"score": None, "justification": result_str, "detailed_feedback": {}}

    # Parse each criterion's result
    feasibility = parse_single_result(evaluation["feasibility"])
    innovation = parse_single_result(evaluation["innovation"])

    return {
        "scores": {
            "feasibility": feasibility["score"],
            "innovation": innovation["score"]
        },
        "justifications": {
            "feasibility": feasibility["justification"],
            "innovation": innovation["justification"]
        },
        "detailed_feedback": {
            "feasibility": feasibility["detailed_feedback"],
            "innovation": innovation["detailed_feedback"]
        }
    }

def get_top_designs(designs_with_scores, limit=3):
    """Returns the top N designs based on total score. In case of ties, later designs are preferred."""
    def calculate_total_score(indexed_entry):
        idx, entry = indexed_entry
        try:
            scores = entry.get('scores', {})
            # Convert any None or non-numeric values to 0
            numeric_scores = [float(score) if score is not None and str(score).replace('.','',1).isdigit() else 0 
                            for score in scores.values()]
            # Return tuple of (score, index) - negative index to prefer later entries
            return (sum(numeric_scores), -idx)
        except (ValueError, TypeError, AttributeError):
            return (0, -idx)
    
    # Use enumerate to track original positions
    sorted_designs = sorted(enumerate(designs_with_scores), key=calculate_total_score, reverse=True)
    # Extract just the designs, discarding the indices
    return [design for _, design in sorted_designs[:limit]]