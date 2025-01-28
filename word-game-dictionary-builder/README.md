# Word Game Dictionary Builder

Tools for building and formatting word dictionaries, optimized for word games and similar applications. Features include length-based filtering, unique letter filtering, and trie structure conversion for efficient prefix lookups.

## Example

I used these to build the latest version of [WordRamp](https://reset.kokutech.com/wordramp-letter-scrambler-browser-game), a letter scrambler that requires you to build words of varying lengths (4>5>6>5>4) which only contain unique letters.

## Components

### Dictionary Formatter ([dictionary_formatter.py](./dictionary_formatter.py))
A versatile tool that processes word lists into game-ready formats. Works with any dictionary source that can be converted to the expected JSON format. Features include filtering words by length, filtering for unique letters (no repeats), and converting to trie structure for efficient prefix lookups.

### Wiktionary Parser ([wiktionary_parser.py](./wiktionary_parser.py))
An example data source parser that extracts English words from Wiktionary XML dumps. You can use this or implement your own parser for different dictionary sources.

## Choosing a Dictionary Source

The dictionary formatter works with any word list that can be converted to its JSON format. Some options to consider:

- **[Wiktionary](https://www.wiktionary.org/)** (implemented example)
  - Pros: Large vocabulary, includes modern terms
  - Cons: May include some questionable entries, XML dumps are large
  - License: Creative Commons Attribution-ShareAlike

- **[WordNet](https://wordnet.princeton.edu/)**
  - Pros: High quality, academic source
  - Cons: More formal vocabulary
  - License: Princeton WordNet License

- **Custom Lists**
  - Create your own curated word list
  - Combine multiple sources
  - Full control over content

## Input/Output Formats

### Input
JSON file mapping words to 1:

```json
{
    "word": 1,
    "another": 1
}
```


### Output
Either filtered dictionary (same format as input) or trie structure:

```json
{
    "w": {
        "o": {
            "r": {
                "d": {
                    "*": 1
                }
            }
        }
    }
}
```


## Requirements
- Python 3.6+
- Standard library only (no external dependencies)