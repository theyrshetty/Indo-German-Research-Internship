Multilingual OCR from PDF using Tesseract and PDF2Image
This script performs Optical Character Recognition (OCR) on a PDF file that contains both English and Hindi text. It uses Tesseract OCR along with pdf2image to convert PDF pages into images and extract text from them.

Prerequisites
Before using this script, ensure you have the following installed:

Tesseract OCR: Download and install Tesseract OCR.

Update the tesseract_cmd path in the script to point to your local Tesseract executable (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe).
Poppler for Windows: Poppler is required to convert PDF files into images.

Download Poppler from this link and extract it. Make sure to set the correct path to poppler_path in the script.
Python Libraries:

Install the required libraries using pip:
bash
Copy code
pip install pytesseract pdf2image
How to Use
Configure Tesseract Path: Ensure that pytesseract.pytesseract.tesseract_cmd is set to the correct path of the Tesseract executable on your machine.

Set TESSDATA_PREFIX: Update the environment variable TESSDATA_PREFIX to the directory where Tesseract stores its language data files (tessdata).

Convert PDF to Text:

The function ocr_on_multilingual_pdf converts a PDF file into images and extracts text from each page. It writes the extracted text into a .txt file, which includes both English and Hindi text.
Example:
To run the script, provide the path to your PDF file and specify the output text file path. Update the variables pdf_path and output_txt_path as required.

python
Copy code
pdf_path = r"Downloads\your_pdf_file.pdf"
output_txt_path = r"Downloads\output_text.txt"
ocr_on_multilingual_pdf(pdf_path, output_txt_path)
The OCR results will be saved in output_txt_path with a breakdown of each page's text.

Parameters
pdf_path: Path to the PDF file you want to process.
output_txt_path: Path to save the OCR results as a text file.
Important Notes
Ensure that the PDF is clear enough for Tesseract to process.
You may need to tweak the Tesseract configuration depending on your document quality and languages.
The language combination lang='hin+eng' tells Tesseract to detect both Hindi and English text.
Error Handling
If the file does not exist or an error occurs during the OCR process, the script will display an error message.

License
