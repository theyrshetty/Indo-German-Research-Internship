# Parliament Document Processing System 

## textextraction.py
Python script to convert PDF documents to text while preserving formatting information.

### Functions:
- `is_hindi_text()`: Detects Hindi characters in text
- `calculate_white_ratio()`: Calculates ratio of white pixels in image region
- `apply_erosion()`: Applies erosion to detect bold text
- `process_page_with_selective_erosion()`: Processes each page with bold detection
- `process_pdf()`: Main function to process entire PDF

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

3. Set input/output files:
   - In textextraction.py:
     ```python
     # Change input PDF path (around line 120)
     pdf_path = "16-III-01.12.2014.pdf"  # Update with your PDF name
     
     # Change output text file name (in process_pdf function parameters)
     def process_pdf(pdf_path, output_file="16-III-01.12.2014.txt"):
     ```
   
   - In tabulation.py:
     ```python
     # Change input text file (in main function)
     processor.process_file('16-III-01.12.2014.txt')  # Update with your text file name
     
     # Output CSV name is automatically generated from input text filename
     # e.g., '16-III-01.12.2014.txt' becomes '16-III-01.12.2014.csv'
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

## File Path Requirements
- All files (PDF, generated text file, and final CSV) should be in the same directory as the Python scripts
- Use consistent file naming across both scripts
- For different input files, update the file names in both scripts accordingly
