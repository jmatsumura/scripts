"""
Assess divergence between the first and last designs of the minimalist and expressive agents.

To be used in conjunction with the output from the `rivals.py` script.
"""

import json
import sys
from sentence_transformers import SentenceTransformer, util

def analyze_design_divergence(json_file):
    # Load and parse JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get first and last designs for each style
    minimalist_designs = data["Minimalist"]["designs"]
    minimalist_first = minimalist_designs[0]
    minimalist_last = minimalist_designs[-1]
    
    expressive_designs = data["Expressive"]["designs"]
    expressive_first = expressive_designs[0]
    expressive_last = expressive_designs[-1]
    
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculate similarities
    min_embedding1 = model.encode(minimalist_first)
    min_embedding2 = model.encode(minimalist_last)
    min_similarity = util.cos_sim(min_embedding1, min_embedding2)
    
    exp_embedding1 = model.encode(expressive_first)
    exp_embedding2 = model.encode(expressive_last)
    exp_similarity = util.cos_sim(exp_embedding1, exp_embedding2)
    
    print("Minimalist design divergence:")
    print("Cosine similarity:", min_similarity.item())
    print("\nExpressive design divergence:")
    print("Cosine similarity:", exp_similarity.item())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python assess-divergence.py <path_to_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    analyze_design_divergence(json_file)