import pytesseract
from pdf2image import convert_from_path
import os
import re
from Levenshtein import distance
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    english_corrections: Dict[str, str]  # Maps incorrect words to their corrections
    hindi_corrections: Dict[str, str]  # Maps incorrect words to their corrections

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
        # Increased threshold for leniency by 15%
        self.threshold = 1.5  # Original threshold was 2, now more lenient
    
    def is_hindi_word(self, word: str) -> bool:
        """Check if a word contains Hindi characters"""
        # Unicode range for Hindi/Devanagari: U+0900 to U+097F
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        return bool(hindi_pattern.search(word))
    
    def is_english_word(self, word: str) -> bool:
        """Check if a word contains only English characters"""
        # Check if the word contains only ASCII letters
        return all(ord(c) < 128 for c in word) and bool(re.match(r'^[a-zA-Z]+$', word))
    
    def calculate_error_percentage(self, text: str) -> Tuple[int, int, float, Dict[str, str], int, int, float, Dict[str, str]]:
        """Calculate error percentage for both English and Hindi text"""
        # First, separate Hindi and English words
        words = re.findall(r'\b\w+\b', text)
        english_words = [w.lower() for w in words if self.is_english_word(w)]
        hindi_words = [w for w in words if self.is_hindi_word(w)]
        
        eng_total = len(english_words)
        hin_total = len(hindi_words)
        
        eng_errors = 0
        hin_errors = 0
        eng_corrections = {}
        hin_corrections = {}
        
        # Process English words
        for word in english_words:
            is_error, correction = self._check_english_word(word)
            if is_error:
                eng_errors += 1
                if correction:
                    eng_corrections[word] = correction
        
        # Process Hindi words
        for word in hindi_words:
            is_error, correction = self._check_hindi_word(word)
            if is_error:
                hin_errors += 1
                if correction:
                    hin_corrections[word] = correction
        
        # Calculate error percentages
        eng_error_percentage = (eng_errors / eng_total * 100) if eng_total > 0 else 0.0
        hin_error_percentage = (hin_errors / hin_total * 100) if hin_total > 0 else 0.0
        
        return eng_total, eng_errors, eng_error_percentage, eng_corrections, hin_total, hin_errors, hin_error_percentage, hin_corrections
    
    def _check_english_word(self, word: str) -> Tuple[bool, Optional[str]]:
        """Check if an English word is considered an error and provide English correction"""
        if word in self.english_dict:
            return False, None
        
        # Find best match in English dictionary only
        min_distance = float('inf')
        best_match = None
        
        sample_size = min(1000, len(self.english_dict))
        sample_dict = list(self.english_dict)[:sample_size]
        
        for dict_word in sample_dict:
            word_distance = distance(word, dict_word)
            if word_distance < min_distance:
                min_distance = word_distance
                best_match = dict_word
        
        # More lenient threshold check
        if min_distance <= self.threshold:
            return False, None  # Not considered an error with the more lenient threshold
        else:
            return True, best_match  # Consider it an error and suggest a correction
    
    def _check_hindi_word(self, word: str) -> Tuple[bool, Optional[str]]:
        """Check if a Hindi word is considered an error and provide Hindi correction"""
        if word in self.hindi_dict:
            return False, None
        
        # Find best match in Hindi dictionary only
        min_distance = float('inf')
        best_match = None
        
        sample_size = min(1000, len(self.hindi_dict))
        sample_dict = list(self.hindi_dict)[:sample_size]
        
        for dict_word in sample_dict:
            word_distance = distance(word, dict_word)
            if word_distance < min_distance:
                min_distance = word_distance
                best_match = dict_word
        
        # More lenient threshold check
        if min_distance <= self.threshold:
            return False, None  # Not considered an error with the more lenient threshold
        else:
            return True, best_match  # Consider it an error and suggest a correction

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
    
    def process_file(self, input_path: str, output_path: str) -> None:
        """Process input file and generate error analysis"""
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            pages = self._read_pages(input_path)
            analysis_results = self._analyze_pages(pages)
            self._write_results(output_path, pages, analysis_results)
            
            logger.info(f"Analysis completed successfully. Results written to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
    
    def _read_pages(self, input_path: Path) -> List[str]:
        """Read and split input file into pages"""
        with input_path.open('r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into pages using regex to handle different page marker formats
        pages = re.split(r'---\s*Page\s+\d+\s*---', content)
        return [page.strip() for page in pages if page.strip()]
    
    def _analyze_pages(self, pages: List[str]) -> List[PageAnalysis]:
        """Analyze each page and return results"""
        results = []
        for i, page_text in enumerate(pages, 1):
            eng_total, eng_errors, eng_error_percentage, eng_corrections, \
            hin_total, hin_errors, hin_error_percentage, hin_corrections = self.analyzer.calculate_error_percentage(page_text)
            
            # Update global error dictionaries
            self.all_english_errors.update(eng_corrections)
            self.all_hindi_errors.update(hin_corrections)
            
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
            
            for result in results:
                f.write(
                    f"{result.page_number} | {result.english_total} | {result.english_errors} | "
                    f"{result.english_error_percentage:.2f} | {result.hindi_total} | {result.hindi_errors} | "
                    f"{result.hindi_error_percentage:.2f}\n"
                )
            
            # Write corrections table for each page
            f.write("\n\n--- Word Corrections by Page ---\n")
            for result in results:
                if result.english_corrections or result.hindi_corrections:
                    f.write(f"\nPage {result.page_number} Corrections:\n")
                    
                    if result.english_corrections:
                        f.write("\nEnglish Corrections:\n")
                        f.write("Incorrect Word | Suggested Correction\n")
                        f.write("-" * 40 + "\n")
                        for word, correction in result.english_corrections.items():
                            f.write(f"{word} | {correction}\n")
                    
                    if result.hindi_corrections:
                        f.write("\nHindi Corrections:\n")
                        f.write("Incorrect Word | Suggested Correction\n")
                        f.write("-" * 40 + "\n")
                        for word, correction in result.hindi_corrections.items():
                            f.write(f"{word} | {correction}\n")
            
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

def main():
    # Configuration
    english_dict_path = "edic.txt"
    hindi_dict_path = "hdic.txt"
    input_path = "errtry.txt"
    output_path = "a.txt"
    
    try:
        detector = OCRErrorDetector(english_dict_path, hindi_dict_path)
        detector.process_file(input_path, output_path)
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()