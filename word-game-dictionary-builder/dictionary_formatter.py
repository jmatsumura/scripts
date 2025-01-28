"""
Dictionary Formatter for Word Games

This script processes dictionary files for word games, with features including:
- Length-based word filtering
- Unique letter word filtering (no repeated letters)
- Trie structure conversion for efficient prefix lookups
- JSON input/output support

Usage:
    python dictionary_formatter.py --lengths 3,4,5 [options]

Options:
    --lengths LENGTHS      Comma-separated list of word lengths to include
    --trie                Convert output to trie structure
    --unique-letters      Only include words with no repeated letters
    --input PATH          Input dictionary JSON file path
    --output PATH         Output JSON file path

The input dictionary should be a JSON file mapping words to values (typically 1).
The output will be either a filtered dictionary or trie structure in JSON format.
"""

import json
import argparse
from typing import Set, Dict

def load_dictionary(json_path: str) -> Dict[str, int]:
    """Load the dictionary from JSON file.
    
    Args:
        json_path: Path to the input JSON dictionary file
        
    Returns:
        Dictionary mapping words to values (typically 1)
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        JSONDecodeError: If input file isn't valid JSON
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def has_unique_letters(word: str) -> bool:
    """Check if word has no repeated letters.
    
    Args:
        word: The word to check
        
    Returns:
        True if the word has no repeated letters, False otherwise
    """
    return len(word) == len(set(word))

def filter_by_lengths(words: Dict[str, int], lengths: Set[int], unique_letters: bool = False) -> Dict[str, int]:
    """Filter words to only include specified lengths and optionally unique letters.
    
    Args:
        words: Dictionary mapping words to values
        lengths: Set of allowed word lengths
        unique_letters: If True, only include words with no repeated letters
        
    Returns:
        Filtered dictionary containing only words matching the criteria
    """
    return {
        word: 1 for word in words 
        if len(word) in lengths 
        and (not unique_letters or has_unique_letters(word))
    }

def convert_to_trie(words: Dict[str, int]) -> Dict:
    """Convert word list to a trie structure for efficient prefix lookups.
    
    Args:
        words: Dictionary mapping words to values
        
    Returns:
        Nested dictionary representing the trie where:
        - Each key is a letter
        - Each value is either another letter dict or '*' marking end of word
        - '*' key with value 1 marks valid word endings
    """
    trie = {}
    for word in words:
        current = trie
        for char in word:
            if char not in current:
                current[char] = {}
            current = current[char]
        current['*'] = 1  # Mark end of word
    return trie

def save_dictionary(data: Dict, output_path: str):
    """Save dictionary to JSON file.
    
    Args:
        data: Dictionary data to save
        output_path: Path to output JSON file
        
    Raises:
        IOError: If unable to write to output path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def main():
    """Main function to parse Wiktionary XML dump and save English words to JSON."""
    parser = argparse.ArgumentParser(description='Format dictionary with length filtering and trie conversion')
    parser.add_argument('--lengths', type=str, required=True,
                      help='Comma-separated list of word lengths to include (e.g., "3,4,5")')
    parser.add_argument('--trie', action='store_true',
                      help='Convert output to trie structure')
    parser.add_argument('--unique-letters', action='store_true',
                      help='Only include words with no repeated letters')
    parser.add_argument('--input', type=str, default='data/wiktionary_words.json',
                      help='Input dictionary JSON file')
    parser.add_argument('--output', type=str, default='data/formatted_dictionary.json',
                      help='Output JSON file')

    args = parser.parse_args()

    # Parse lengths
    try:
        lengths = {int(l.strip()) for l in args.lengths.split(',')}
    except ValueError:
        print("Error: Lengths must be comma-separated numbers")
        return

    print(f"Loading dictionary from {args.input}...")
    words = load_dictionary(args.input)
    
    print(f"Filtering words to lengths: {sorted(lengths)}")
    if args.unique_letters:
        print("Filtering to words with unique letters only")
    filtered_words = filter_by_lengths(words, lengths, args.unique_letters)
    
    if args.trie:
        print("Converting to trie structure...")
        output_data = convert_to_trie(filtered_words)
    else:
        output_data = filtered_words
    
    print(f"Saving to {args.output}...")
    save_dictionary(output_data, args.output)
    
    print(f"Done! Processed {len(filtered_words)} words")

if __name__ == "__main__":
    main() 