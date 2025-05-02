import pytesseract
from pdf2image import convert_from_path
import os
import re
from Levenshtein import distance
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
import logging
from dataclasses import dataclass
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WordPosition(NamedTuple):
    """Store position information for a word"""
    page: int
    line: int
    word: int
    text: str  # The context text

@dataclass
class PageAnalysis:
    """Data class to store analysis results for a single page"""
    page_number: int
    english_total: int
    english_errors: int
    english_error_percentage: float
    hindi_total: int
    hindi_errors: int
    hindi_error_percentage: float
    english_corrections: Dict[str, Tuple[str, WordPosition]]  # Maps incorrect words to (correction, position)
    hindi_corrections: Dict[str, Tuple[str, WordPosition]]  # Maps incorrect words to (correction, position)

class DictionaryLoader:
    """Handles dictionary loading and caching"""
    @staticmethod
    def load_dictionary(file_path: str) -> Set[str]:
        """Load dictionary from file and return as set of words"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Dictionary file not found: {file_path}")
            
            with path.open('r', encoding='utf-8') as f:
                return {word.strip().lower() for word in f if word.strip()}
        except Exception as e:
            logger.error(f"Error loading dictionary {file_path}: {str(e)}")
            raise

class TextAnalyzer:
    """Handles text analysis and error detection"""
    def __init__(self, english_dict: Set[str], hindi_dict: Set[str]):
        self.english_dict = english_dict
        self.hindi_dict = hindi_dict
        # Different thresholds for English and Hindi
        self.english_threshold = 1
        self.hindi_threshold = 2.5  # More lenient threshold for Hindi
    
    def is_hindi_word(self, word: str) -> bool:
        """Check if a word contains Hindi characters"""
        # Expanded Unicode range for Hindi/Devanagari: U+0900 to U+097F
        # Also include Vedic extensions, Devanagari Extended
        hindi_pattern = re.compile(r'[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]')
        return bool(hindi_pattern.search(word))
    
    def is_english_word(self, word: str) -> bool:
        """Check if a word contains only English characters"""
        # Check if the word contains only ASCII letters
        return all(ord(c) < 128 for c in word) and bool(re.match(r'^[a-zA-Z]+$', word))
    
    def calculate_error_percentage(self, text: str, page_num: int) -> Tuple[int, int, float, Dict[str, Tuple[str, WordPosition]], int, int, float, Dict[str, Tuple[str, WordPosition]]]:
        """Calculate error percentage for both English and Hindi text with position tracking"""
        lines = text.split('\n')
        
        eng_total = 0
        hin_total = 0
        eng_errors = 0
        hin_errors = 0
        eng_corrections = {}
        hin_corrections = {}
        
        for line_num, line in enumerate(lines, 1):
            # Improved word extraction
            words = re.findall(r'\b[\w\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+\b', line)
            
            # Process words in this line
            word_num = 0
            for word in words:
                word_num += 1
                
                # Get context (up to 50 chars before and after the word)
                word_pos = line.find(word)
                start_pos = max(0, word_pos - 50)
                end_pos = min(len(line), word_pos + len(word) + 50)
                context = line[start_pos:end_pos]
                
                position = WordPosition(page=page_num, line=line_num, word=word_num, text=context)
                
                if self.is_english_word(word):
                    eng_total += 1
                    is_error, correction = self._check_english_word(word.lower())
                    if is_error:
                        eng_errors += 1
                        if correction:
                            eng_corrections[word.lower()] = (correction, position)
                
                elif self.is_hindi_word(word):
                    hin_total += 1
                    is_error, correction = self._check_hindi_word(word)
                    if is_error:
                        hin_errors += 1
                        if correction:
                            hin_corrections[word] = (correction, position)
                
                else:
                    # Try to classify mixed words based on character majority
                    hindi_chars = sum(1 for c in word if '\u0900' <= c <= '\u097F' or '\u1CD0' <= c <= '\u1CFF' or '\uA8E0' <= c <= '\uA8FF')
                    english_chars = sum(1 for c in word if 'a' <= c.lower() <= 'z')
                    
                    if hindi_chars > english_chars:
                        hin_total += 1
                        is_error, correction = self._check_hindi_word(word)
                        if is_error:
                            hin_errors += 1
                            if correction:
                                hin_corrections[word] = (correction, position)
                    elif english_chars > hindi_chars:
                        eng_total += 1
                        is_error, correction = self._check_english_word(word.lower())
                        if is_error:
                            eng_errors += 1
                            if correction:
                                eng_corrections[word.lower()] = (correction, position)
        
        # Calculate error percentages
        eng_error_percentage = (eng_errors / eng_total * 100) if eng_total > 0 else 0.0
        hin_error_percentage = (hin_errors / hin_total * 100) if hin_total > 0 else 0.0
        
        return eng_total, eng_errors, eng_error_percentage, eng_corrections, hin_total, hin_errors, hin_error_percentage, hin_corrections
    
    def _check_english_word(self, word: str) -> Tuple[bool, Optional[str]]:
        """Check if an English word is considered an error and provide English correction"""
        if word in self.english_dict:
            return False, None
        
        # Find best match in English dictionary
        min_distance = float('inf')
        best_match = None
        
        # Improved sampling - use more words for smaller dictionaries
        sample_size = min(3000, len(self.english_dict))
        sample_dict = list(self.english_dict)[:sample_size]
        
        # For short words, check the entire dictionary
        if len(word) <= 3:
            sample_dict = self.english_dict
        
        for dict_word in sample_dict:
            # Skip dictionary words with very different lengths
            if abs(len(dict_word) - len(word)) > 3:
                continue
                
            word_distance = distance(word, dict_word)
            if word_distance < min_distance:
                min_distance = word_distance
                best_match = dict_word
        
        # Adjusted threshold based on word length
        adaptive_threshold = min(self.english_threshold, len(word) * 0.4)
        if min_distance <= adaptive_threshold:
            return False, None
        else:
            return True, best_match
    
    def _check_hindi_word(self, word: str) -> Tuple[bool, Optional[str]]:
        """Check if a Hindi word is considered an error and provide Hindi correction"""
        if word in self.hindi_dict:
            return False, None
        
        # Find best match in Hindi dictionary
        min_distance = float('inf')
        best_match = None
        
        # Use larger sample size for Hindi
        sample_size = min(5000, len(self.hindi_dict))
        sample_dict = list(self.hindi_dict)[:sample_size]
        
        # For short words, check the entire dictionary
        if len(word) <= 3:
            sample_dict = self.hindi_dict
        
        for dict_word in sample_dict:
            # Skip dictionary words with very different lengths
            if abs(len(dict_word) - len(word)) > 3:
                continue
                
            # Calculate normalized distance for Hindi
            word_distance = distance(word, dict_word)
            normalized_distance = word_distance / max(len(word), len(dict_word))
            
            if normalized_distance < min_distance:
                min_distance = normalized_distance
                best_match = dict_word
        
        # More lenient threshold for Hindi and adaptive to word length
        adaptive_threshold = min(0.5, (self.hindi_threshold / 10))  # Normalize to 0-1 range
        
        if min_distance <= adaptive_threshold:
            return False, None
        else:
            return True, best_match

class OCRErrorDetector:
    """Main class for OCR error detection"""
    def __init__(self, english_dict_path: str, hindi_dict_path: str):
        self.dict_loader = DictionaryLoader()
        self.english_dict = self.dict_loader.load_dictionary(english_dict_path)
        self.hindi_dict = self.dict_loader.load_dictionary(hindi_dict_path)
        self.analyzer = TextAnalyzer(self.english_dict, self.hindi_dict)
        # Track all errors across pages
        self.all_english_errors = {}
        self.all_hindi_errors = {}
    
    def process_file(self, input_path: str, output_path: str, corrected_output_path: str) -> None:
        """Process input file and generate error analysis and corrected text"""
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            corrected_output_path = Path(corrected_output_path)
            
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            pages = self._read_pages(input_path)
            analysis_results = self._analyze_pages(pages)
            
            # Write analysis report
            self._write_results(output_path, pages, analysis_results)
            
            # Write corrected text
            self._write_corrected_text(corrected_output_path, pages, analysis_results)
            
            logger.info(f"Analysis completed successfully. Results written to {output_path}")
            logger.info(f"Corrected text written to {corrected_output_path}")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
    
    def _read_pages(self, input_path: Path) -> List[str]:
        """Read and split input file into pages"""
        with input_path.open('r', encoding='utf-8') as f:
            content = f.read()
        
        # Improved page splitting - handles various page marker formats
        pages = re.split(r'---\s*Page\s+\d+\s*---', content)
        # Handle case where there might not be page markers
        if len(pages) <= 1 and len(content) > 500:
            # If no page markers but content is long, split into pages of roughly equal size
            approx_page_size = 2000  # characters
            pages = [content[i:i+approx_page_size] for i in range(0, len(content), approx_page_size)]
        
        return [page.strip() for page in pages if page.strip()]
    
    def _analyze_pages(self, pages: List[str]) -> List[PageAnalysis]:
        """Analyze each page and return results"""
        results = []
        for i, page_text in enumerate(pages, 1):
            eng_total, eng_errors, eng_error_percentage, eng_corrections, \
            hin_total, hin_errors, hin_error_percentage, hin_corrections = self.analyzer.calculate_error_percentage(page_text, i)
            
            # Update global error dictionaries - store just the corrections without positions
            for word, (correction, _) in eng_corrections.items():
                self.all_english_errors[word] = correction
            
            for word, (correction, _) in hin_corrections.items():
                self.all_hindi_errors[word] = correction
            
            results.append(PageAnalysis(
                page_number=i,
                english_total=eng_total,
                english_errors=eng_errors,
                english_error_percentage=eng_error_percentage,
                hindi_total=hin_total,
                hindi_errors=hin_errors,
                hindi_error_percentage=hin_error_percentage,
                english_corrections=eng_corrections,
                hindi_corrections=hin_corrections
            ))
        
        return results
    
    def _write_results(self, output_path: Path, pages: List[str], results: List[PageAnalysis]) -> None:
        """Write analysis results to output file"""
        with output_path.open('w', encoding='utf-8') as f:
            # Write processed pages
            for i, (page, analysis) in enumerate(zip(pages, results), 1):
                f.write(f"--- Page {i} ---\n{page}\n\n")
            
            # Write summary table
            f.write("\n--- Summary ---\n")
            f.write("Page | English Words | English Errors | English Error % | Hindi Words | Hindi Errors | Hindi Error %\n")
            f.write("-" * 80 + "\n")
            
            total_english_words = 0
            total_english_errors = 0
            total_hindi_words = 0
            total_hindi_errors = 0
            
            for result in results:
                total_english_words += result.english_total
                total_english_errors += result.english_errors
                total_hindi_words += result.hindi_total
                total_hindi_errors += result.hindi_errors
                
                f.write(
                    f"{result.page_number} | {result.english_total} | {result.english_errors} | "
                    f"{result.english_error_percentage:.2f}% | {result.hindi_total} | {result.hindi_errors} | "
                    f"{result.hindi_error_percentage:.2f}%\n"
                )
            
            # Calculate overall error percentages
            overall_eng_percent = (total_english_errors / total_english_words * 100) if total_english_words > 0 else 0
            overall_hindi_percent = (total_hindi_errors / total_hindi_words * 100) if total_hindi_words > 0 else 0
            
            # Write overall statistics
            f.write("-" * 80 + "\n")
            f.write(
                f"Total | {total_english_words} | {total_english_errors} | "
                f"{overall_eng_percent:.2f}% | {total_hindi_words} | {total_hindi_errors} | "
                f"{overall_hindi_percent:.2f}%\n"
            )
            
            # Write corrections table for each page with position information
            f.write("\n\n--- Word Corrections by Page ---\n")
            for result in results:
                if result.english_corrections or result.hindi_corrections:
                    f.write(f"\nPage {result.page_number} Corrections:\n")
                    
                    if result.english_corrections:
                        f.write("\nEnglish Corrections:\n")
                        f.write("Incorrect Word | Suggested Correction | Line | Word | Context\n")
                        f.write("-" * 100 + "\n")
                        for word, (correction, position) in result.english_corrections.items():
                            f.write(f"{word} | {correction} | {position.line} | {position.word} | {position.text}\n")
                    
                    if result.hindi_corrections:
                        f.write("\nHindi Corrections:\n")
                        f.write("Incorrect Word | Suggested Correction | Line | Word | Context\n")
                        f.write("-" * 100 + "\n")
                        for word, (correction, position) in result.hindi_corrections.items():
                            f.write(f"{word} | {correction} | {position.line} | {position.word} | {position.text}\n")
            
            # Write complete list of all errors
            f.write("\n\n--- Complete List of Incorrect Words ---\n")
            
            if self.all_english_errors:
                f.write("\nEnglish Incorrect Words:\n")
                f.write("Incorrect Word | Suggested Correction\n")
                f.write("-" * 40 + "\n")
                for word, correction in sorted(self.all_english_errors.items()):
                    f.write(f"{word} | {correction}\n")
            
            if self.all_hindi_errors:
                f.write("\nHindi Incorrect Words:\n")
                f.write("Incorrect Word | Suggested Correction\n")
                f.write("-" * 40 + "\n")
                for word, correction in sorted(self.all_hindi_errors.items()):
                    f.write(f"{word} | {correction}\n")
    
    def _write_corrected_text(self, output_path: Path, pages: List[str], results: List[PageAnalysis]) -> None:
        """Write corrected version of text to output file"""
        with output_path.open('w', encoding='utf-8') as f:
            for i, (page, analysis) in enumerate(zip(pages, results), 1):
                f.write(f"--- Page {i} ---\n")
                
                # Create a combined corrections dictionary for the page
                corrections = {}
                for word, (correction, _) in analysis.english_corrections.items():
                    corrections[word] = correction
                for word, (correction, _) in analysis.hindi_corrections.items():
                    corrections[word] = correction
                
                # Apply corrections
                corrected_page = self._apply_corrections(page, corrections)
                f.write(f"{corrected_page}\n\n")
    
    def _apply_corrections(self, text: str, corrections: Dict[str, str]) -> str:
        """Apply corrections to text"""
        if not corrections:
            return text
        
        # Create a function for the re.sub replacement
        def replace_word(match):
            word = match.group(0)
            # Try lowercase for English words
            if word.lower() in corrections:
                return corrections[word.lower()]
            # Try exact match for Hindi words
            elif word in corrections:
                return corrections[word]
            else:
                return word
        
        # Build a regex pattern that matches all the words to be corrected
        # Sort by length (longest first) to avoid partial matches
        words_to_correct = sorted(corrections.keys(), key=len, reverse=True)
        
        # Escape special regex characters in words
        escaped_words = [re.escape(word) for word in words_to_correct]
        
        # Create pattern
        pattern = r'\b(?:' + '|'.join(escaped_words) + r')\b'
        
        # Apply replacements
        corrected_text = re.sub(pattern, replace_word, text, flags=re.IGNORECASE)
        
        return corrected_text

def process_pdf_to_text(pdf_path: str, output_text_path: str) -> None:
    """Convert PDF to text using pytesseract OCR"""
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        with open(output_text_path, 'w', encoding='utf-8') as text_file:
            for i, image in enumerate(images, 1):
                # Perform OCR with support for Hindi
                text = pytesseract.image_to_string(image, lang='eng+hin')
                
                # Write to file with page marker
                text_file.write(f"--- Page {i} ---\n{text}\n\n")
                
        logger.info(f"PDF conversion completed. Text written to {output_text_path}")
        
    except Exception as e:
        logger.error(f"Error converting PDF to text: {str(e)}")
        raise

def main():
    # Configuration
    english_dict_path = "edic.txt"
    hindi_dict_path = "hi_IN.dic"
    input_path = "errtry.txt"
    output_path = "analysis_report.txt"
    corrected_output_path = "corrected_text.txt"
    
    # Optional PDF processing
    pdf_mode = False
    pdf_path = "document.pdf"
    text_output_path = "extracted_text.txt"
    
    try:
        if pdf_mode:
            # Step 1: Convert PDF to text
            process_pdf_to_text(pdf_path, text_output_path)
            input_path = text_output_path
        
        # Step 2: Analyze text for errors and generate corrected version
        detector = OCRErrorDetector(english_dict_path, hindi_dict_path)
        detector.process_file(input_path, output_path, corrected_output_path)
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
