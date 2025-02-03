"""
Generate similarity matrices between all designs for both Systematic and Whimsical agents.
Outputs CSV files that can be used for visualization.

To be used in conjunction with the output from the `gamedev-rivals.py` script.
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

def save_matrix_to_csv(matrix, output_file, is_cross=False, num_systematic=0):
    """Save matrix to CSV file with clear labeling.
    
    Args:
        matrix: The similarity matrix to save
        output_file: Path to save the CSV
        is_cross: Whether this is the cross-philosophy matrix
        num_systematic: Number of systematic designs (used for cross-matrix labeling)
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Create headers with appropriate labels
        if is_cross:
            # For cross matrix, label which indices are Systematic vs Whimsical
            systematic_labels = [f"Systematic_{i}" for i in range(num_systematic)]
            whimsical_labels = [f"Whimsical_{i}" for i in range(len(matrix) - num_systematic)]
            header = [''] + systematic_labels + whimsical_labels
        else:
            # For single philosophy matrices, just use numerical indices
            header = [''] + [str(i) for i in range(len(matrix))]
            
        writer.writerow(header)
        
        # Write each row with appropriate row labels
        for i, row in enumerate(matrix):
            rounded_row = [round(x, 3) for x in row]
            if is_cross:
                row_label = f"Systematic_{i}" if i < num_systematic else f"Whimsical_{i-num_systematic}"
            else:
                row_label = str(i)
            writer.writerow([row_label] + rounded_row)

def analyze_design_matrices(json_file):
    # Load and parse JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get designs for each style, removing the think blocks
    systematic_designs = [design.split("</think>")[1].strip() if "</think>" in design else design.strip() 
                        for design in data["Systematic"]["designs"]]
    whimsical_designs = [design.split("</think>")[1].strip() if "</think>" in design else design.strip() 
                        for design in data["Whimsical"]["designs"]]
    
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculate similarity matrices
    sys_matrix = create_similarity_matrix(systematic_designs, model)
    whim_matrix = create_similarity_matrix(whimsical_designs, model)
    
    # Calculate cross-philosophy similarity
    all_designs = systematic_designs + whimsical_designs
    cross_matrix = create_similarity_matrix(all_designs, model)
    
    # Save to CSV files
    base_name = json_file.replace('.json', '')
    sys_output = f"{base_name}_systematic_similarity.csv"
    whim_output = f"{base_name}_whimsical_similarity.csv"
    cross_output = f"{base_name}_cross_philosophy_similarity.csv"
    
    save_matrix_to_csv(sys_matrix, sys_output)
    save_matrix_to_csv(whim_matrix, whim_output)
    save_matrix_to_csv(cross_matrix, cross_output, is_cross=True, num_systematic=len(systematic_designs))
    
    print(f"Systematic similarity matrix saved to: {sys_output}")
    print(f"Whimsical similarity matrix saved to: {whim_output}")
    print(f"Cross-philosophy similarity matrix saved to: {cross_output}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python assess-divergence.py <path_to_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    analyze_design_matrices(json_file)