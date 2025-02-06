import argparse
import json
import csv
from pathlib import Path

def parse_json_file(file_path):
    """Parse a single creator-critic JSON results file into a list of records."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    records = []
    for i, design in enumerate(data['designs'], 1):
        scores = design['scores']
        record = {
            'total_score': scores['innovation'] + scores['feasibility'],
            'innovation_score': scores['innovation'],
            'feasibility_score': scores['feasibility'],
            'iteration': i
        }
        records.append(record)
    
    return records

def main():
    parser = argparse.ArgumentParser(description='Parse creator-critic JSON results into CSV')
    parser.add_argument('--files', nargs='+', required=True,
                       help='List of JSON files to parse')
    parser.add_argument('--labels', nargs='+', required=True,
                       help='Labels corresponding to each file')
    parser.add_argument('--output', default='results.csv',
                       help='Output CSV file name (default: results.csv)')
    
    args = parser.parse_args()
    
    if len(args.files) != len(args.labels):
        raise ValueError("Number of files must match number of labels")
    
    all_records = []
    
    # Process each file
    for file_path, label in zip(args.files, args.labels):
        records = parse_json_file(file_path)
        for record in records:
            record['label'] = label
        all_records.extend(records)
    
    # Write to CSV
    fieldnames = ['total_score', 'innovation_score', 'feasibility_score', 'iteration', 'label']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 