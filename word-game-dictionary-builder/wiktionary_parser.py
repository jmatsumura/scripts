"""
Wiktionary XML Dump Parser

This script extracts English words from a Wiktionary XML dump file.
It filters entries to include only valid English words, excluding meta pages,
phrases, acronyms, and non-English entries.

The parser produces a JSON dictionary mapping valid English words to 1,
suitable for use with the dictionary_formatter.py script.

Usage:
    python wiktionary_parser.py [--input XML_PATH] [--output JSON_PATH]

Input:
    Wiktionary XML dump file (e.g., enwiktionary-latest-pages-articles.xml)
    Download from: https://dumps.wikimedia.org/enwiktionary/latest/

Output:
    JSON file mapping words to 1 (e.g., {"word": 1, "another": 1})

Word Filtering Rules:
    - Must have an English section
    - Must contain only ASCII alphabetic characters
    - No spaces or colons in title
    - No acronyms or all-caps entries
    - Excludes meta pages, categories, templates, etc.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import time
import sys
import json
import argparse

def is_valid_english_entry(title: str, text: str) -> bool:
    """Check if this is a valid English word entry.
    
    Args:
        title: The page title from Wiktionary
        text: The full page content
        
    Returns:
        True if entry is a valid English word, False otherwise
        
    The function applies multiple filters to ensure the entry is:
    - Not a meta page (categories, templates, etc.)
    - Contains only ASCII alphabetic characters
    - Not an acronym or all-caps entry
    - Has an English language section
    """
    # Skip meta pages, categories, templates, etc.
    if any(title.startswith(prefix) for prefix in [
        "Wiktionary:", "Template:", "Category:", 
        "Help:", "Appendix:", "Concordance:", 
        "Thesaurus:", "Rhymes:", "Index:", 
        "MediaWiki:", "Citations:", "Sign gloss:",
        "Reconstruction:"
    ]):
        return False
        
    # Skip entries with colons or spaces (usually phrases or meta entries)
    if ":" in title or " " in title:
        return False

    # Skip acronyms (all caps or multiple consecutive capitals)
    if title.isupper() or any(
        title[i].isupper() and title[i+1].isupper() 
        for i in range(len(title)-1)
    ):
        return False

    # Only accept words containing purely ASCII alphabetic characters
    try:
        if not title.encode('ascii').decode('ascii').isalpha():
            return False
    except UnicodeEncodeError:
        return False
    
    # Check if it has an English section
    if not "==English==" in text:
        return False


    return True


def parse_wiktionary_dump(xml_path, output_path):
    """Parse Wiktionary XML dump and extract English word entries.
    
    Args:
        xml_path: Path to the Wiktionary XML dump file
        output_path: Path where the JSON output should be saved
        
    Returns:
        Number of unique English words extracted
        
    Raises:
        FileNotFoundError: If XML file doesn't exist
        ValueError: If XML file is empty
        ET.ParseError: If XML parsing fails
        
    The function streams the XML to minimize memory usage and provides
    progress updates every 25,000 entries processed.
    """
    entry_count = 0
    words_dict = {}
    start_time = time.time()
    
    print(f"Opening XML file: {xml_path}")
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"Input file not found: {xml_path}")
    
    namespace = "http://www.mediawiki.org/xml/export-0.11/"
    
    try:
        print("Starting XML parsing...")
        context = ET.iterparse(xml_path, events=('end',))
        
        try:
            _, root = next(context)
            print("Successfully got root element")
        except StopIteration:
            raise ValueError("XML file appears to be empty")
        
        for event, elem in context:
            if elem.tag.endswith('page'):
                # Get both title and text content
                title = elem.find(f'.//{{{namespace}}}title')
                revision = elem.find(f'.//{{{namespace}}}revision')
                text = revision.find(f'.//{{{namespace}}}text') if revision is not None else None
                
                if title is not None and text is not None:
                    title_text = title.text
                    content_text = text.text or ""
                    
                    # Only process valid English entries
                    if (is_valid_english_entry(title_text, content_text)):
                        # Store word in lowercase, using dict for deduplication
                        word = title_text.lower()
                        words_dict[word] = True
                        entry_count += 1
                        
                        if entry_count % 25000 == 0:
                            elapsed_time = time.time() - start_time
                            rate = entry_count / elapsed_time
                            print(f"Processed {entry_count:,} English entries... ({rate:.1f} entries/sec)")
                
                # Clear element to save memory
                elem.clear()
                root.clear()
        
        # Write the final dictionary to JSON
        print("\nWriting words to JSON file...")
        with open(output_path, 'w', encoding='utf-8') as out_file:
            # Convert dict keys to sorted list and write as JSON object
            word_list = sorted(words_dict.keys())
            json_dict = {word: 1 for word in word_list}
            json.dump(json_dict, out_file, indent=2)
        
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    
    return len(words_dict)

def main():
    """Main function to parse Wiktionary XML dump and save English words to JSON."""
    parser = argparse.ArgumentParser(description='Parse Wiktionary XML dump for English words')
    parser.add_argument('--input', '-i', 
                      default="data/enwiktionary-latest-pages-articles.xml",
                      help='Path to input XML file')
    parser.add_argument('--output', '-o', 
                      default="data/wiktionary_words.json",
                      help='Path to output JSON file')
    
    args = parser.parse_args()
    
    print(f"Starting to parse Wiktionary dump from {args.input}...")
    print(f"Words will be saved to {args.output}")
    
    try:
        start_time = time.time()
        total_entries = parse_wiktionary_dump(args.input, args.output)
        
        elapsed_time = time.time() - start_time
        print(f"\nParsing complete in {elapsed_time:.1f} seconds")
        print(f"Extracted {total_entries:,} unique English words")
        print("Done!")
    except Exception as e:
        print(f"Failed to process file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 