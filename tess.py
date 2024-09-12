import pytesseract
from pdf2image import convert_from_path
import os

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set the TESSDATA_PREFIX environment variable to the tessdata directory
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Function to perform OCR on Hindi PDF and print the text
def ocr_on_hindi_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    try:
        # Convert the PDF to images (each page as an image)
        images = convert_from_path(pdf_path, poppler_path=r"C:\Users\Daksh Vats\Downloads\Release-24.07.0-0\poppler-24.07.0\Library\bin")
        
        # Iterate over each page image
        for i, image in enumerate(images):
            # Perform OCR on the image (Hindi language)
            text = pytesseract.image_to_string(image, lang='hin')
            
            # Print the OCR text for each page
            print(f"--- Page {i + 1} ---")
            print(text)
            print("\n")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
pdf_path = r"Downloads\16-IX-05.08.2016.pdf"
ocr_on_hindi_pdf(pdf_path)
