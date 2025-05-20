# Parliament Document Processing System 

## text-extraction.py
Python script to convert PDF documents to text while preserving formatting information.

### Functions:
- `is_speaker_formatted()`: Checks if a word has speaker-specific formatting (bold or capitalized).
- `validate_speaker()`: Validates and extracts the speaker's name from the text.
- `process_line()`: Processes each line of text based on the current processing mode.
- `process_init_mode()`: Handles lines in the initial mode, looking for session markers.
- `process_page_mode()`: Extracts page numbers and metadata from the lines.
- `process_potential_speaker_mode()`: Manages multi-line speaker names and validates potential speaker lines.
- `save_current_record()`: Saves the current speaker and speech data to the records.
- `process_file()`: Processes the entire input file and generates a structured CSV output.


### Features:
- Uses erosion technique for bold text detection
  - Converts text regions to binary image
  - Applies erosion using 4x4 kernel
  - Calculates white-to-black pixel ratio to determine boldness
- Enhances contrast of scanned pages
- Preserves line breaks based on vertical positioning
- Supports both Hindi and English text processing
- Outputs formatted text with bold markers (**) for Hindi text

## tabulation.py
Python script to convert processed text into structured CSV data.

### Functions:
- `is_speaker_formatted()`: Checks text formatting (bold/caps)
- `validate_speaker_line()`: Validates speaker identification
- `process_line()`: Processes individual lines of text
- `process_file()`: Main function to generate CSV output
- `save_current_record()`: Saves processed data to records

### Features:
- Identifies speakers based on text formatting
- Tracks page numbers and metadata
- Handles multi-line speaker names
- Processes speeches with appropriate attribution
- Exports structured data to CSV with page, metadata, speaker, and speech columns

## name_sourcing.py 

Python script to match MP names from official lists with speaker names in parliamentary speech data.

### Functions:
* `normalize_name()`: Standardizes name formatting by converting to lowercase, removing punctuation, and extra spaces
* `is_speaker_chair()`: Identifies if a speaker name refers to the Speaker/Chair of the House
* `is_chair_chair()`: Identifies if a speaker name refers to the Chairperson
* `normalize_hindi_name()`: Creates both standard and no-space versions of Hindi names
* `check_name_words_match()`: Compares speaker name with MP name through word-by-word matching
* `calculate_string_similarity()`: Calculates string similarity as a backup metric
* `find_top_matches()`: Identifies top 2 matching MP names for each speaker
* `match_mp_names()`: Main function to process and match names across datasets

### Features:
* Name normalization - Converts text to lowercase, removes punctuation and excess spaces, with special handling for Hindi names to account for space variations.

* Multi-strategy name matching - Employs word-by-word matching, sequential word matching with score boosts, and string similarity calculations for more accurate results.

* Hindi name recognition - Handles combined word forms, substring matching for names with prefixes, and calculates match quality based on proportion of matching characters.

* Parliamentary role handling - Detects and standardizes Speaker/Chair references in both English and Hindi, assigning consistent designations for these roles.

* Score-based ranking - Calculates scores using weighted combinations of word matching and string similarity, returning the top two most likely MP matches.

## name_matching.py 

Python script to determine the correct MP name match from multiple preferences based on contextual analysis.

### Functions:
* `process_csv()`: Main function to process matched names and determine correct speaker
* `is_hindi()`: Detects if text contains Hindi language
* `get_alternating_pattern()`: Identifies A-B-A-B dialogue patterns

### Features:
* Context-aware name resolution - Identifies top 4 most occurring names and applies special handling for single-occurrence names that might represent errors.

* Decision logic - Employs three specialized logical paths to determine the most probable correct speaker:
  * Path 1: Single-occurrence resolution - Selects preference 2 when a name appears once in preference 1 but preference 2 contains a frequent name.
  * Path 2: Hindi speaker analysis - When original text is Hindi, checks if preference 2 name is prominent and frequently appears in English text rows; if both true, selects preference 2, otherwise uses preference 1.
  * Path 3: Pattern-based correction - Selects preference 2 if it maintains an established dialogue pattern that preference 1 would break.

* Dialogue pattern recognition - Analyzes speaker sequences to identify A-B-A-B patterns, tracking position to predict the next expected speaker.

* Comprehensive fallbacks - Implements cascading fallbacks from preferred matching to alternatives, ensuring no records have empty speaker fields.

## Required Python Packages
```
pytesseract==0.3.10
pdf2image==1.16.3
Pillow==9.5.0
opencv-python==4.8.0
numpy==1.24.3
pandas==2.0.3
```


## Running the System

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Update paths in textextraction.py:
```python
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Path to Tesseract
poppler_path = r'C:\Program Files\poppler-24.07.0\Library\bin'    # Path to Poppler
```

3. Set input files:
   - In textextraction.py:
     ```python
     # Change input PDF path (in main function)
     pdf_path = "16-III-01.12.2014.pdf"  # Update with your PDF name
     # Output .txt file is automatically generated with same name as PDF
     ```
   
   - In tabulation.py:
     ```python
     # Change input text file (in main function)
     processor.process_file('16-III-01.12.2014.txt')  # Update with your text file name
     # Output CSV name is automatically generated from input text filename
     ```

4. Run in sequence:
```bash
python textextraction.py  # Creates .txt file
python tabulation.py     # Creates .csv file
```



## Configuration Options

### textextraction.py
- Adjust contrast: Modify `enhance(2)` value
- Bold detection threshold: Change `bold_threshold = 0.915`
- Line break sensitivity: Adjust `line_height_threshold = 10`

### tabulation.py
- Speaker validation: Modify `non_speaker_words <= 3`
- Speaker formatting: Update `is_speaker_formatted()` conditions
- 
## File Path Requirements
- All files (PDF, generated text file, and final CSV) should be in the same directory as the Python scripts
- File names are handled automatically:
  - textextraction.py: converts 'filename.pdf' to 'filename.txt'
  - tabulation.py: converts 'filename.txt' to 'filename.csv'
