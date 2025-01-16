import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os

# Define paths
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
poppler_path = r'C:\Program Files\poppler-24.07.0\Library\bin'

# Set Tesseract path and update environment
pytesseract.pytesseract.tesseract_cmd = tesseract_path
os.environ['PATH'] += os.pathsep + poppler_path

def is_hindi_text(text):
    """Check if text contains Hindi characters."""
    return any("\u0900" <= c <= "\u097F" for c in text)

def calculate_white_ratio(image_region):
    """
    Calculate the ratio of white pixels to total pixels in an image region.
    Returns a value between 0 and 1, where 1 means all pixels are white.
    """
    if image_region.mode != 'L':
        image_region = image_region.convert('L')
    
    image_array = np.array(image_region)
    white_threshold = 200
    white_pixels = np.sum(image_array > white_threshold)
    total_pixels = image_array.size
    
    return white_pixels / total_pixels

def apply_erosion(image_region):
    """Apply erosion to reduce text thickness."""
    # Convert to grayscale if not already
    if image_region.mode != 'L':
        grayscale_image = image_region.convert("L")
    else:
        grayscale_image = image_region
    
    image_array = np.array(grayscale_image)
    
    # Create binary image
    _, binary_image = cv2.threshold(image_array, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Define erosion kernel
    kernel = np.ones((4, 4), np.uint8)
    
    # Apply erosion
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    
    # Invert back
    eroded_image_inverted = cv2.bitwise_not(eroded_image)
    
    return Image.fromarray(eroded_image_inverted)

def process_page_with_selective_erosion(page_image, config="--psm 6 --oem 3 -l eng+hin"):
    """
    Process a single page, applying erosion only to Hindi text for bold detection.
    Includes improved line break detection using vertical positions.
    """
    # First pass: Get all words and their locations
    data = pytesseract.image_to_data(page_image, config=config, output_type=pytesseract.Output.DICT)
    
    # Initialize output text and tracking
    extracted_text = ""
    current_line = data['line_num'][0] if data['line_num'] else None
    last_y_bottom = None
    line_height_threshold = 10  # Adjust this value based on your PDF's characteristics
    
    # Process each word
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        
        # Get bounding box coordinates
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        current_y_bottom = y + h
        
        # Check for new line based on both line_num and vertical position
        is_new_line = False
        if last_y_bottom is not None:
            # Check if there's a significant vertical gap
            vertical_gap = y - (last_y_bottom)
            if vertical_gap > line_height_threshold:
                is_new_line = True
        
        if data['line_num'][i] != current_line or is_new_line:
            extracted_text += "\n"
            current_line = data['line_num'][i]
        
        # Extract word region and process Hindi text
        word_region = page_image.crop((x, y, x + w, y + h))
        
        if is_hindi_text(text):
            eroded_region = apply_erosion(word_region)
            white_ratio = calculate_white_ratio(eroded_region)
            bold_threshold = 0.915 #adjust this threshold as per the boldness
            if white_ratio < bold_threshold:
                text = f"**{text}**"
        
        extracted_text += f"{text} "
        last_y_bottom = current_y_bottom
    
    return extracted_text.strip()

def process_pdf(pdf_path):
    """
    Process entire PDF and save formatted output.
    """
    output_file = os.path.splitext(pdf_path)[0] + '.txt'
    
    print(f"Processing PDF: {pdf_path}")
    
    # Convert PDF to images
    pages = convert_from_path(pdf_path, 300)
    formatted_output = ""
    
    for page_number, page in enumerate(pages, 1):
        print(f"Processing page {page_number}...")
        
        # Convert to RGB and enhance contrast
        image = page.convert('RGB')
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(2)
        
        # Process page
        page_text = process_page_with_selective_erosion(enhanced_image)
        formatted_output += f"Page {page_number}:\n{page_text}\n\n"
    
    # Save output
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(formatted_output)
    
    print(f"Processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    pdf_path = "16-III-01.12.2014.pdf"  # Update with your PDF path
    process_pdf(pdf_path)
