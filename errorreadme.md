# OCR Error Detector

A robust Python tool for detecting and analyzing OCR (Optical Character Recognition) errors in bilingual text documents supporting English and Hindi. This tool uses Levenshtein distance and dictionary-based validation to identify potential OCR errors and generate detailed analysis reports.

## Features

- Bilingual text analysis (English and Hindi support)
- Page-by-page error detection and analysis
- Detailed error reporting with statistical summaries
- Configurable error threshold using Levenshtein distance
- Comprehensive logging system
- Dictionary-based word validation

## Key Components

### DictionaryLoader

Handles the loading and caching of dictionary files for both English and Hindi languages. The dictionaries are used as reference sets for word validation.

```python
dictionary = DictionaryLoader.load_dictionary("words_alpha.txt")
```

### TextAnalyzer

Performs the core text analysis using:
- Word tokenization
- Levenshtein distance calculation
- Error percentage computation
- Dictionary-based validation

The analyzer uses a configurable threshold for Levenshtein distance to determine if a word is considered an error.

### PageAnalysis

A data class that stores analysis results for each page, including:
- Total word count (English and Hindi)
- Error count
- Error percentage
- Page number

### OCRErrorDetector

The main class that orchestrates the entire error detection process:
1. Loads dictionaries
2. Processes input files
3. Analyzes text content
4. Generates detailed reports

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- pytesseract
- pdf2image
- python-Levenshtein

## Usage

1. Prepare your dictionary files:
   - English dictionary: `words_alpha.txt`
   - Hindi dictionary: `hi_IN.dic`

2. Run the detector:

```python
from ocr_error_detector import OCRErrorDetector

detector = OCRErrorDetector("words_alpha.txt", "hi_IN.dic")
detector.process_file("input.txt", "output.txt")
```

## Input Format

The input file should contain text with page markers in the format:
```
--- Page 1 ---
[page content]
--- Page 2 ---
[page content]
```

## Output Format

The tool generates a detailed report containing:
1. Processed text content for each page
2. Summary table with statistics:
   - Page number
   - Word counts (English/Hindi)
   - Error counts
   - Error percentages

Example output:
```
--- Page 1 ---
[processed content]

--- Summary ---
Page | English Words | English Errors | English Error % | Hindi Words | Hindi Errors | Hindi Error %
--------------------------------------------------------------------------------
1    | 100          | 5             | 5.00           | 50          | 2            | 4.00
```

## Error Detection Method

The tool uses Levenshtein distance to measure the similarity between words:
1. Words are extracted using regex pattern `\b\w+\b`
2. Each word is compared against the appropriate dictionary
3. If a word isn't found in the dictionary, it's compared against a sample of dictionary words
4. Words with a Levenshtein distance greater than the threshold (default: 2) are marked as errors

## Performance Optimization

- Dictionary sampling for Levenshtein distance calculation
- Efficient file handling using Pathlib
- Memory-efficient text processing

## Logging

The tool includes comprehensive logging:
- Log level: INFO
- Format: timestamp - level - message
- Error tracking for dictionary loading and file processing

## Error Handling

Robust error handling for:
- File not found errors
- Dictionary loading issues
- Text processing errors
- Invalid input formats
(Levenshtein Distance: This is a metric used to measure how different two strings are by counting the minimum number of single-character edits required to change one string into another. In this code, it's used to detect potential OCR errors by comparing words against dictionary entries.
Dictionary-based Validation: The system uses two dictionaries (English and Hindi) as reference sets. Words not found in these dictionaries are considered potential errors, but the system also checks for similar words using Levenshtein distance to account for minor OCR mistakes.
Data Classes: The code uses Python's @dataclass decorator for the PageAnalysis class, which automatically generates special methods like __init__ and __repr__ based on the class attributes. This makes the code more concise and less prone to errors.
Type Hints: The code uses Python's type hinting system (e.g., List[str], Set[str]) to make the code more maintainable and easier to debug. This helps catch type-related errors early in development.
Logging System: The code implements a comprehensive logging system that tracks important events and errors during execution. This is crucial for debugging and monitoring the application's behavior.
Performance Optimization: The code includes several optimizations:

Dictionary sampling for Levenshtein distance calculations
Efficient file handling using pathlib
Set data structures for O(1) dictionary lookups


Error Handling: The code implements robust error handling using try-except blocks and custom error messages, ensuring graceful failure handling and useful error reporting.)
