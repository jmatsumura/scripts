"""
Generate similarity matrices between all designs for both minimalist and expressive agents.
Outputs CSV files that can be used for visualization.

To be used in conjunction with the output from the `rivals.py` script.
"""

import json
import sys
import csv
from sentence_transformers import SentenceTransformer, util

def create_similarity_matrix(designs, model):
    """Calculate similarity matrix for a list of designs."""
    # Encode all designs at once for efficiency
    embeddings = model.encode(designs)
    
    # Calculate similarity matrix
    similarity_matrix = util.cos_sim(embeddings, embeddings)
    
    # Convert to regular Python list of lists
    return similarity_matrix.tolist()

def save_matrix_to_csv(matrix, output_file):
    """Save matrix to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row with indices
        header = [''] + [str(i) for i in range(len(matrix))]
        writer.writerow(header)
        
        # Write each row with row index, rounding each value to 6 decimal places
        for i, row in enumerate(matrix):
            rounded_row = [round(x, 3) for x in row]
            writer.writerow([str(i)] + rounded_row)

def analyze_design_matrices(json_file):
    # Load and parse JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get designs for each style
    minimalist_designs = data["Minimalist"]["designs"]
    expressive_designs = data["Expressive"]["designs"]
    
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculate similarity matrices
    min_matrix = create_similarity_matrix(minimalist_designs, model)
    exp_matrix = create_similarity_matrix(expressive_designs, model)
    
    # Save to CSV files
    base_name = json_file.replace('.json', '')
    min_output = f"{base_name}_minimalist_similarity.csv"
    exp_output = f"{base_name}_expressive_similarity.csv"
    
    save_matrix_to_csv(min_matrix, min_output)
    save_matrix_to_csv(exp_matrix, exp_output)
    
    print(f"Minimalist similarity matrix saved to: {min_output}")
    print(f"Expressive similarity matrix saved to: {exp_output}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python assess-design-matrix.py <path_to_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    analyze_design_matrices(json_file) 