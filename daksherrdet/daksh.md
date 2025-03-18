# OCR Error Detection System

A Python tool for detecting and correcting errors in OCR-processed text, with special support for bilingual (English-Hindi) documents.

## Overview

This system analyzes text extracted from documents via OCR (Optical Character Recognition) to identify potential errors in both English and Hindi text. It compares words against dictionaries, calculates error percentages, and suggests corrections based on Levenshtein distance.

## Features

- Bilingual support for **English** and **Hindi** text analysis
- Page-by-page error detection and statistics
- Detailed error reports with suggested corrections
- Adjustable tolerance threshold for error detection
- Support for PDF processing via PDF to image conversion

## Prerequisites

- Python 3.6+
- Required Python packages:
  - pytesseract
  - pdf2image
  - python-Levenshtein
  - poppler (for PDF processing)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ocr-error-detector.git
   cd ocr-error-detector
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR and Poppler:
   - **Windows**: Download and install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](http://blog.alivate.com.au/poppler-windows/)
   - **Mac**: `brew install tesseract poppler`
   - **Linux**: `sudo apt-get install tesseract-ocr poppler-utils`

4. Prepare your dictionary files:
   - Place your English dictionary in `edic.txt`
   - Place your Hindi dictionary in `hdic.txt`

## Usage

### Basic Usage

```python
from ocr_error_detector import OCRErrorDetector

detector = OCRErrorDetector("edic.txt", "hdic.txt")
detector.process_file("input_ocr_text.txt", "analysis_results.txt")
```

### Command Line

```
python main.py
```

### Configuration

Edit the main function in the script to configure:
- Dictionary paths
- Input file path
- Output file path

```python
def main():
    # Configuration
    english_dict_path = "edic.txt"
    hindi_dict_path = "hdic.txt"
    input_path = "errtry.txt"
    output_path = "a.txt"
    
    # ...
```

## Input Format

The input file should contain OCR-extracted text with page markers in the format:
```
--- Page 1 ---
[Page 1 content]

--- Page 2 ---
[Page 2 content]
```

## Output Format

The analysis output includes:
1. The original text split by pages
2. A summary table with error statistics for each page
3. Word correction suggestions for each page
4. A complete list of all detected errors

## Advanced Features

### Adjusting Error Tolerance

The system uses a threshold value to determine when a word is considered an error. This can be adjusted in the `TextAnalyzer` class:

```python
def __init__(self, english_dict: Set[str], hindi_dict: Set[str]):
    # ...
    self.threshold = 1.5  # Lower values = stricter, Higher values = more lenient
```

### PDF Processing

To process PDF files directly:

```python
def process_pdf(self, pdf_path: str, output_path: str) -> None:
    pages = convert_from_path(pdf_path)
    all_text = []
    
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang='eng+hin')
        all_text.append(f"--- Page {i+1} ---\n{text}")
    
    with open('temp_ocr_output.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    
    self.process_file('temp_ocr_output.txt', output_path)
    os.remove('temp_ocr_output.txt')
```

## Architecture

The system consists of three main components:

1. **DictionaryLoader**: Handles loading and caching of dictionaries
2. **TextAnalyzer**: Performs text analysis and error detection
3. **OCRErrorDetector**: Main class that orchestrates the process

## License

[MIT License](LICENSE)

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-new-feature`
5. Submit a pull request

## Authors

- Daksh Vats - *Initial work*
