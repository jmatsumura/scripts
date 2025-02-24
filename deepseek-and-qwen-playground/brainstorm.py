from dataclasses import dataclass
from typing import Dict, Optional, List, ClassVar
import os
import requests
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import json
from datetime import datetime

# ===== Prompt Templates =====

# Example Context (customize for your use case)
PROBLEM_STATEMENT_CONTEXT = "What kind of algorithm could I use to draw pixel art?"
# example2: "I need a title for my new movie set in a cyberpunk dystopia"
# example3: "What is the best way for me to learn piano?"

# Branch Identification Prompt
BRANCH_IDENTIFICATION_PROMPT = """# CORE PROBLEM STATEMENT
{context}

# GUIDELINES
Analyze the thought below and identify up to three distinct branches that directly address this specific problem.
For each branch, provide:
1. A title describing the solution approach
2. The key elements to explore that directly relate to solving the original problem
3. Why this approach would help solve the specific problem posed

## OUTPUT FORMAT
BRANCH: <solution approach>
CONCEPT: <key elements>
REASON: <why this helps solve the original problem>

# YOUR TASK
Identify the three most promising branches that stay strictly within the scope of the original problem."""

# Thinking Constraints Template
THINKING_CONSTRAINTS = """## THINKING CONSTRAINTS
- Identify 2 complementary approaches
- Find 1 paradoxical element
- Suggest 3 concrete adjustments"""

INSTRUCTION_GUIDE = """## INSTRUCTION GUIDE
- Create step-by-step recommendations
- Keep advice practical
- Include implementation tips"""

# ===== Configurable Parameters =====

# Model Configuration
## on CLI run
### ollama run deepseek-r1:32b
### ollama run qwen2.5:32b
MODEL_CONFIG = {
    "reasoning_model": "deepseek-r1:32b",  # Model for generating thoughts
    "instruction_model": "qwen2.5:32b",    # Model for processing and insights
    "ollama_endpoint": "http://localhost:11434",
    "sentence_transformer": "all-MiniLM-L6-v2"  # Model for similarity checking
}

# Generation Parameters
GENERATION_CONFIG = {
    "temperature_reasoning": 0.8,    # Higher = more creative thoughts
    "temperature_instruction": 0.7,  # Lower = more focused insights
    "max_branch_depth": 5,          # How deep to explore thought branches
    "similarity_threshold": 0.6      # Higher = less similar thoughts allowed
}

# Combine configs for internal use
DEFAULT_CONFIG = {**MODEL_CONFIG, **GENERATION_CONFIG}

# Initialize sentence transformer model
sentence_model = SentenceTransformer(MODEL_CONFIG["sentence_transformer"])

@dataclass
class CognitiveSeed:
    """Container for the reasoning model's output with branching support"""
    raw_thoughts: str
    processed_thoughts: str
    parent_thought: Optional['CognitiveSeed'] = None
    branch_depth: int = 0
    thought_history: List[str] = None
    branch_title: str = None  # Track which branch this thought represents
    branch_concept: str = None  # Store the branch concept
    children: List['CognitiveSeed'] = None
    metadata: Optional[Dict] = None
    # Class variable to track all branches across the tree
    _all_branches: ClassVar[List[Dict[str, str]]] = []
    
    def __post_init__(self):
        if self.thought_history is None:
            self.thought_history = []
        if self.children is None:
            self.children = []
        self.thought_history = self.thought_history + [self.processed_thoughts]
        # Add branch info to global list if this is a branch node
        if self.branch_title and self.branch_concept:
            CognitiveSeed._all_branches.append({
                'title': self.branch_title,
                'concept': self.branch_concept
            })
    
    @classmethod
    def reset_branches(cls):
        """Reset the global branch list - call before starting a new tree"""
        cls._all_branches = []

def compute_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts using sentence transformers."""
    # Encode texts to get embeddings
    embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def check_seed_similarity(seed: CognitiveSeed, config: Dict) -> bool:
    """Check if a CognitiveSeed is too similar to any previously explored branches."""
    if not seed.branch_title or not seed.branch_concept:
        return False
        
    threshold = config["similarity_threshold"]
    branch_text = f"{seed.branch_title}\n{seed.branch_concept}"
    
    # Check against all previously explored branches
    for existing_branch in CognitiveSeed._all_branches:
        existing_text = f"{existing_branch['title']}\n{existing_branch['concept']}"
        similarity = compute_similarity(branch_text, existing_text)
        if similarity > threshold:
            return True
    
    return False

def check_branch_similarity(branch: Dict[str, str], config: Dict) -> bool:
    """Check if a branch dictionary is too similar to any previously explored branches."""
    if not branch.get('title') or not branch.get('concept'):
        return False
        
    threshold = config["similarity_threshold"]
    branch_text = f"{branch['title']}\n{branch['concept']}"
    
    # Check against all previously explored branches
    for existing_branch in CognitiveSeed._all_branches:
        existing_text = f"{existing_branch['title']}\n{existing_branch['concept']}"
        similarity = compute_similarity(branch_text, existing_text)
        if similarity > threshold:
            return True
    
    return False

def query_ollama(prompt: str, config: Dict, model: str, system_prompt: str = None) -> str:
    """Generic Ollama query handler"""
    temperature = config["temperature_reasoning"] if model == config["reasoning_model"] else config["temperature_instruction"]

    if os.getenv("PROMPT_DEBUG"):
        print(f"🤖 Querying {model} with prompt:\n{prompt}")
        if system_prompt:
            print(f"🤖 System prompt:\n{system_prompt}")
    
    if system_prompt:
        # Use chat API with system prompt
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        response = requests.post(f"{config['ollama_endpoint']}/api/chat", json=payload)
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
    else:
        # Use generate API without system prompt
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        response = requests.post(f"{config['ollama_endpoint']}/api/generate", json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
    
    return f"Error: {response.status_code} - {response.text}"

def extract_think_block(text: str) -> str:
    """Extract the content between <think> tags."""
    if "<think>" in text and "</think>" in text:
        return text.split("<think>")[1].split("</think>")[0].strip()
    return text

def generate_thoughts(
    context: str,
    constraints: str,
    parent_seed: Optional[CognitiveSeed] = None,
    config: Dict = DEFAULT_CONFIG
) -> CognitiveSeed:
    """
    Generates raw cognitive patterns using reasoning model with branching support
    """
    branch_depth = 0 if parent_seed is None else parent_seed.branch_depth + 1
    thought_history = [] if parent_seed is None else parent_seed.thought_history

    print(f"{' ' * branch_depth}🤔 Generating thoughts at branch depth {branch_depth}...")

    if branch_depth >= config["max_branch_depth"]:
        print("  ⚠️  Maximum branch depth reached")
        return None

    # Construct prompt context, prepending root problem if context differs
    prompt_context = context
    if context != PROBLEM_STATEMENT_CONTEXT:
        prompt_context = f"""# ROOT PROBLEM TO BE SOLVED
{PROBLEM_STATEMENT_CONTEXT}

# CURRENT CONTEXT
{context}"""
    else:
        prompt_context = f"""# PROBLEM STATEMENT
{context}"""

    prompt = f"""{prompt_context}

{constraints}

Think deeply about the problem and generate a new perspective or solution that directly addresses it.
If exploring a branch, ensure your solution ties back to the root problem."""

    raw_output = query_ollama(prompt, config, config["reasoning_model"])
    
    # Extract think block and output separately
    thoughts = extract_think_block(raw_output)
    output = raw_output.split("</think>")[-1].strip() if "</think>" in raw_output else raw_output
    
    # Create the seed first so we can use it for similarity checking
    seed = CognitiveSeed(
        raw_thoughts=thoughts,
        processed_thoughts=output,
        parent_thought=parent_seed,
        branch_depth=branch_depth,
        thought_history=thought_history,
        metadata={"branch_level": branch_depth}
    )
    
    # Check if this thought path is worth exploring
    if parent_seed and check_seed_similarity(seed, config):
        return None
        
    return seed

def process_thoughts(seed: CognitiveSeed) -> CognitiveSeed:
    """
    Transforms raw thoughts by removing note lines
    """
    processed = "\n".join([
        line for line in seed.raw_thoughts.split("\n")
        if not line.strip().startswith("Note:")
    ])
    return CognitiveSeed(
        raw_thoughts=seed.raw_thoughts,
        processed_thoughts=processed,
        metadata={"processed": True}
    )

def identify_branches(seed: CognitiveSeed, config: Dict) -> List[Dict]:
    """Analyze a thought to identify distinct conceptual branches."""
    system_prompt = """You are a thought analyzer. Your role is to identify distinct and promising branches of thought that could be explored further, while staying strictly within the bounds of the original problem context."""
    
    response = query_ollama(
        prompt=f"{BRANCH_IDENTIFICATION_PROMPT.format(context=PROBLEM_STATEMENT_CONTEXT)}\n\n# THOUGHT:\n{seed.processed_thoughts}",
        config=config,
        model=config["instruction_model"],
        system_prompt=system_prompt
    )
    
    # Parse the response into branch dictionaries
    branches = []
    current_branch = {}
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('BRANCH:'):
            if current_branch:
                branches.append(current_branch)
            current_branch = {'title': line[7:].strip()}
        elif line.startswith('CONCEPT:'):
            current_branch['concept'] = line[8:].strip()
        elif line.startswith('REASON:'):
            current_branch['reason'] = line[7:].strip()
    
    if current_branch:
        branches.append(current_branch)
    
    return branches

def explore_thought_tree(
    context: str,
    seed: CognitiveSeed,
    constraints: str,
    config: Dict = DEFAULT_CONFIG
) -> CognitiveSeed:
    """Recursively explore branches of thought."""
    visited_branches = set()
    
    # Reset global branch tracking when starting a new tree
    if seed.branch_depth == 0:
        CognitiveSeed.reset_branches()
    
    if seed.branch_depth >= config["max_branch_depth"]:
        return seed
    
    print(f"{' ' * seed.branch_depth}🌳 Exploring thought branches at depth {seed.branch_depth}...")
    
    # Identify potential branches from this thought
    branches = identify_branches(seed, config)
    print(f"{' ' * seed.branch_depth}📍 Found {len(branches)} potential branches to explore")
    
    for branch in branches:
        # Check if this branch is too similar to previously explored ones
        if check_branch_similarity(branch, config):
            print(f"{' ' * seed.branch_depth}⚠️  Skipping similar branch: {branch['title']}")
            continue
            
        branch_key = f"{branch['title']}_{seed.branch_depth}"
        if branch_key in visited_branches:
            continue
            
        visited_branches.add(branch_key)
        print(f"{' ' * seed.branch_depth}🔍 Exploring branch: {branch['title']}")
        
        # Generate a new thought for this branch
        parent_context = f"Branching from: {seed.branch_title}" if seed.branch_title else "Initial exploration"
        
        branch_prompt = f"""Explore this specific aspect:
        
        ## Parent Branch
        {parent_context}
        
        ## Branch to Explore
        {branch['title']}

        ## Concept to explore
        {branch['concept']}"""
        
        new_seed = generate_thoughts(
            context=branch_prompt,
            constraints=constraints,
            parent_seed=seed,
            config=config
        )
        
        if new_seed:
            new_seed.branch_title = branch['title']
            new_seed.branch_concept = branch['concept']
            seed.children.append(new_seed)
            # Recursively explore this branch
            explore_thought_tree(context, new_seed, constraints, config)
    
    return seed

def process_thought_tree(root_seed: CognitiveSeed) -> List[Dict]:
    """Process all thoughts in the tree and return structured branch insights."""
    insights = []
    print("🔄 Processing thought tree...")
    
    def traverse(seed: CognitiveSeed):
        if seed:
            processed = process_thoughts(seed)
            if processed:
                insight = {
                    'depth': seed.branch_depth,
                    'branch_title': seed.branch_title,
                    'branch_concept': seed.branch_concept,
                    'thoughts': processed.processed_thoughts
                }
                insights.append(insight)
                print(f"  ✓ Processed branch at depth {seed.branch_depth}" + (f" ({seed.branch_title})" if seed.branch_title else ""))
            for child in seed.children:
                traverse(child)
    
    traverse(root_seed)
    return insights

def serialize_thought_tree(seed: CognitiveSeed) -> Dict:
    """Convert a thought tree into a serializable dictionary."""
    if not seed:
        return None
        
    result = {
        'raw_thoughts': seed.raw_thoughts,
        'processed_thoughts': seed.processed_thoughts,
        'branch_depth': seed.branch_depth,
        'branch_title': seed.branch_title,
        'thought_history': seed.thought_history,
        'metadata': seed.metadata,
        'children': [serialize_thought_tree(child) for child in seed.children] if seed.children else []
    }
    return result

def save_results(thought_tree: CognitiveSeed, branch_insights: List[Dict], context: str):
    """Save the thought exploration results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"brainstorm_{timestamp}.json"
    sift_filename = f"brainstorm_{timestamp}_sift.json"
    
    # Extract just the branch concepts for simplified format
    simplified_insights = []
    for insight in branch_insights:
        if insight.get('branch_concept'):
            simplified_insights.append(insight['branch_concept'])
    
    # Main detailed results
    results = {
        'context': context,
        'timestamp': datetime.now().isoformat(),
        'thought_tree': serialize_thought_tree(thought_tree),
        'branch_insights': branch_insights,
        'config': {
            'model_config': MODEL_CONFIG,
            'generation_config': GENERATION_CONFIG,
            'thinking_constraints': THINKING_CONSTRAINTS
        }
    }
    
    # Simplified insights
    sift_results = {
        'insights': simplified_insights
    }
    
    print(f"\n💾 Saving results to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saving concept list to {sift_filename}...")
    with open(sift_filename, 'w', encoding='utf-8') as f:
        json.dump(sift_results, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Results saved successfully")

# Example usage pattern
if __name__ == "__main__":
    print("🚀 Starting thought exploration process...")
    
    # Generate initial seed thought
    print("📝 Generating initial seed thought...")
    initial_seed = generate_thoughts(
        context=PROBLEM_STATEMENT_CONTEXT,
        constraints=THINKING_CONSTRAINTS
    )
    
    # Explore the thought tree
    print("🌱 Beginning thought tree exploration...")
    thought_tree = explore_thought_tree(
        context=PROBLEM_STATEMENT_CONTEXT,
        seed=initial_seed,
        constraints=THINKING_CONSTRAINTS
    )
    
    # Process all thoughts in the tree
    print("📊 Processing thoughts by branch...")
    branch_insights = process_thought_tree(thought_tree)
    
    # Save results to JSON file
    save_results(thought_tree, branch_insights, PROBLEM_STATEMENT_CONTEXT)
    
    print("\n✅ Process complete!")