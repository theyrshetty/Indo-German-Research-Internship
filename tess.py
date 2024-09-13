import pytesseract
from pdf2image import convert_from_path
import os

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set the TESSDATA_PREFIX environment variable to the tessdata directory
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Function to perform OCR on a PDF with both English and Hindi text
def ocr_on_multilingual_pdf(pdf_path, output_txt_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    try:
        # Convert the PDF to images (each page as an image)
        images = convert_from_path(pdf_path, poppler_path=r"C:\Users\Daksh Vats\Downloads\Release-24.07.0-0\poppler-24.07.0\Library\bin")
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:  # Open file in write mode
            # Iterate over each page image
            for i, image in enumerate(images):
                # Perform OCR on the image (specifying both Hindi and English languages)
                text = pytesseract.image_to_string(image, lang='hin+eng')  # Specify both Hindi (hin) and English (eng)
                
                # Write the OCR text to the file
                f.write(f"--- Page {i + 1} ---\n")
                f.write(text)
                f.write("\n\n")  # Add spacing between pages
                
        print(f"OCR results saved to {output_txt_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
pdf_path = r"Downloads\16-IX-05.08.2016.pdf"
output_txt_path = r"Downloads\multilingual_ocr_output.txt"
ocr_on_multilingual_pdf(pdf_path, output_txt_path)
