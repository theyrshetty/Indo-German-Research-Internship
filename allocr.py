import pytesseract
from pdf2image import convert_from_path
import os

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set the TESSDATA_PREFIX environment variable to the tessdata directory
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Function to perform OCR on a single PDF with both English and Hindi text
def ocr_on_pdf(pdf_path, output_txt_path):
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
        print(f"An error occurred with file {pdf_path}: {e}")

# Function to perform OCR on all PDFs in the Downloads folder
def ocr_on_all_pdfs_in_folder(folder_path):
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            output_txt_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_ocr_output.txt")
            print(f"Processing {filename}...")
            ocr_on_pdf(pdf_path, output_txt_path)

# Example usage
downloads_folder = r"Downloads"
ocr_on_all_pdfs_in_folder(downloads_folder)
