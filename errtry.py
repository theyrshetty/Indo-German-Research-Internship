import pytesseract
from pdf2image import convert_from_path
import os
import re
from Levenshtein import distance
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging
from dataclasses import dataclass

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
    
    def calculate_error_percentage(self, text: str, language_dict: Set[str]) -> Tuple[int, int, float]:
        """Calculate error percentage using Levenshtein distance"""
        words = re.findall(r'\b\w+\b', text.lower())
        total_words = len(words)
        if not total_words:
            return 0, 0, 0.0
        
        errors = sum(1 for word in words if self._is_error(word, language_dict))
        error_percentage = (errors / total_words) * 100
        
        return total_words, errors, error_percentage
    
    def _is_error(self, word: str, language_dict: Set[str], threshold: int = 2) -> bool:
        """Check if a word is considered an error based on Levenshtein distance"""
        if word in language_dict:
            return False
        
        # Only check a sample of dictionary words for performance
        sample_size = min(1000, len(language_dict))
        sample_dict = set(list(language_dict)[:sample_size])
        
        return min((distance(word, dict_word) for dict_word in sample_dict), default=threshold + 1) > threshold

class OCRErrorDetector:
    """Main class for OCR error detection"""
    def __init__(self, english_dict_path: str, hindi_dict_path: str):
        self.dict_loader = DictionaryLoader()
        self.english_dict = self.dict_loader.load_dictionary(english_dict_path)
        self.hindi_dict = self.dict_loader.load_dictionary(hindi_dict_path)
        self.analyzer = TextAnalyzer(self.english_dict, self.hindi_dict)
    
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
            eng_total, eng_errors, eng_error_percentage = self.analyzer.calculate_error_percentage(
                page_text, self.english_dict
            )
            hin_total, hin_errors, hin_error_percentage = self.analyzer.calculate_error_percentage(
                page_text, self.hindi_dict
            )
            
            results.append(PageAnalysis(
                page_number=i,
                english_total=eng_total,
                english_errors=eng_errors,
                english_error_percentage=eng_error_percentage,
                hindi_total=hin_total,
                hindi_errors=hin_errors,
                hindi_error_percentage=hin_error_percentage
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

def main():
    # Configuration
    english_dict_path = "words_alpha.txt"
    hindi_dict_path = "hi_IN.dic"
    input_path = "errtry.txt"
    output_path = "errtrytable.txt"
    
    try:
        detector = OCRErrorDetector(english_dict_path, hindi_dict_path)
        detector.process_file(input_path, output_path)
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()