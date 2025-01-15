[Previous sections remain the same until "Running the System"]

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

[Rest of the README remains the same]

## File Path Requirements
- All files (PDF, generated text file, and final CSV) should be in the same directory as the Python scripts
- File names are handled automatically:
  - textextraction.py: converts 'filename.pdf' to 'filename.txt'
  - tabulation.py: converts 'filename.txt' to 'filename.csv'
- Only need to specify the input PDF name in textextraction.py
