import pytesseract
from pdf2image import convert_from_path
import os
import re
import csv
from Levenshtein import distance
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, NamedTuple, Counter
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
    row: int
    col: int
    text: str  # The context text

@dataclass
class RowAnalysis:
    """Data class to store analysis results for a single row"""
    row_number: int
    primary_language: str
    total_words: int
    total_errors: int
    error_percentage: float
    corrections: Dict[str, Tuple[str, str, float, WordPosition]]  # Maps incorrect words to (correction1, correction2, distance, position)

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
    
    @staticmethod
    def load_external_words(file_path: str) -> Set[str]:
        """Load external custom words"""
        try:
            path = Path(file_path)
            if path.exists():
                with path.open('r', encoding='utf-8') as f:
                    return {word.strip() for word in f if word.strip()}
            else:
                logger.warning(f"External words file not found: {file_path}. Creating empty file.")
                with path.open('w', encoding='utf-8'):
                    pass
                return set()
        except Exception as e:
            logger.error(f"Error loading external words {file_path}: {str(e)}")
            return set()

class ProperNounTracker:
    """Tracks potential proper nouns and manages their dictionary"""
    def __init__(self, threshold: int = 3):
        self.candidates = defaultdict(int)
        self.threshold = threshold
        self.proper_nouns = set()
    
    def add_candidate(self, word: str) -> None:
        """Add a word as a proper noun candidate"""
        self.candidates[word] += 1
        if self.candidates[word] >= self.threshold:
            self.proper_nouns.add(word)
    
    def is_proper_noun(self, word: str) -> bool:
        """Check if a word is recognized as a proper noun"""
        return word in self.proper_nouns
    
    def save_to_file(self, file_path: str) -> None:
        """Save proper nouns to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for word in sorted(self.proper_nouns):
                    f.write(f"{word}\n")
                
                f.write("\n# Candidates below threshold:\n")
                for word, count in sorted(self.candidates.items()):
                    if count < self.threshold and count > 1:
                        f.write(f"{word} (count: {count})\n")
                        
            logger.info(f"Saved {len(self.proper_nouns)} proper nouns to {file_path}")
        except Exception as e:
            logger.error(f"Error saving proper nouns to {file_path}: {str(e)}")

class TextAnalyzer:
    """Handles text analysis and error detection"""
    def __init__(self, english_dict: Set[str], hindi_dict: Set[str], 
                 english_external: Set[str], hindi_external: Set[str],
                 proper_noun_tracker: ProperNounTracker):
        self.english_dict = english_dict.union(english_external)
        self.hindi_dict = hindi_dict.union(hindi_external)
        self.proper_noun_tracker = proper_noun_tracker
        # Different thresholds for English and Hindi
        self.english_threshold = 2
        self.hindi_threshold = 3  # More lenient threshold for Hindi
        # Proper noun detection threshold
        self.proper_noun_distance_threshold = 4
    
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
    
    def determine_primary_language(self, text: str) -> str:
        """Determine the primary language (English or Hindi) of the text"""
        hindi_words = 0
        english_words = 0
        
        # Extract all words from text
        words = re.findall(r'\b[\w\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+\b', text)
        
        for word in words:
            if self.is_hindi_word(word):
                hindi_words += 1
            elif self.is_english_word(word):
                english_words += 1
        
        return "Hindi" if hindi_words >= english_words else "English"
    
    def _check_language_segment(self, segment: str) -> str:
        """Check if a segment contains more than 5 continuous letters of a specific language"""
        # Check for 5+ continuous Hindi characters
        hindi_sequence = re.search(r'[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]{5,}', segment)
        if hindi_sequence:
            return "Hindi"
        
        # Check for 5+ continuous English characters
        english_sequence = re.search(r'[a-zA-Z]{5,}', segment)
        if english_sequence:
            return "English"
        
        # If no long sequence found, return empty string
        return ""
    
    def analyze_text(self, text: str, row_num: int, primary_language: str = None) -> RowAnalysis:
        """Analyze text for errors based on primary language detection"""
        # Determine primary language if not provided
        if primary_language is None:
            primary_language = self.determine_primary_language(text)
        
        total_words = 0
        total_errors = 0
        corrections = {}
        
        # Extract words from text
        words = re.findall(r'\b[\w\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+\b', text)
        
        for col_num, word in enumerate(words, 1):
            context = self._get_context(text, word)
            position = WordPosition(row=row_num, col=col_num, text=context)
            
            # Determine the language of this specific word/segment
            word_language = ""
            if len(word) >= 5:  # Only check long enough words
                word_language = self._check_language_segment(word)
            
            # If we couldn't determine language from segment length, use primary language
            if not word_language:
                word_language = primary_language
            
            # Process the word based on its detected language
            total_words += 1
            
            # Check if proper noun first (for English words)
            if word_language == "English" and word[0].isupper() and len(word) > 1:
                # Might be a proper noun
                if self.proper_noun_tracker.is_proper_noun(word):
                    continue  # Skip checking as it's a known proper noun
            
            # Check word based on language
            is_error, correction1, correction2, error_distance = self._check_word(word, word_language)
            
            if is_error:
                # Special handling for potential proper nouns (for English)
                if word_language == "English" and word[0].isupper() and error_distance > self.proper_noun_distance_threshold:
                    self.proper_noun_tracker.add_candidate(word)
                    # Still count as error for now until it reaches threshold
                
                total_errors += 1
                corrections[word] = (correction1, correction2, error_distance, position)
        
        # Calculate error percentage
        error_percentage = (total_errors / total_words * 100) if total_words > 0 else 0.0
        
        return RowAnalysis(
            row_number=row_num,
            primary_language=primary_language,
            total_words=total_words,
            total_errors=total_errors,
            error_percentage=error_percentage,
            corrections=corrections
        )
    
    def _get_context(self, text: str, word: str) -> str:
        """Get the context around a word (up to 50 chars before and after)"""
        word_pos = text.find(word)
        if word_pos == -1:
            return ""
            
        start_pos = max(0, word_pos - 50)
        end_pos = min(len(text), word_pos + len(word) + 50)
        return text[start_pos:end_pos]
    
    def _check_word(self, word: str, language: str) -> Tuple[bool, Optional[str], Optional[str], float]:
        """Check if a word is considered an error and provide corrections"""
        if language == "English":
            return self._check_english_word(word)
        else:  # Hindi
            return self._check_hindi_word(word)
    
    def _check_english_word(self, word: str) -> Tuple[bool, Optional[str], Optional[str], float]:
        """Check if an English word is considered an error and provide English corrections"""
        # Check if word is in dictionary (case-insensitive)
        if word.lower() in self.english_dict:
            return False, None, None, 0.0
        
        # Find best matches in English dictionary
        matches = []
        
        # Improved sampling - use more words for smaller dictionaries
        sample_size = min(5000, len(self.english_dict))
        sample_dict = list(self.english_dict)[:sample_size]
        
        # For short words, check the entire dictionary
        if len(word) <= 3:
            sample_dict = self.english_dict
        
        for dict_word in sample_dict:
            # Skip dictionary words with very different lengths
            if abs(len(dict_word) - len(word)) > 3:
                continue
                
            word_distance = distance(word.lower(), dict_word)
            matches.append((dict_word, word_distance))
        
        # Sort by distance
        matches.sort(key=lambda x: x[1])
        
        # Get top two matches if available
        correction1 = matches[0][0] if matches else None
        correction2 = matches[1][0] if len(matches) > 1 else None
        
        # Get the best distance
        best_distance = matches[0][1] if matches else float('inf')
        
        # Adjusted threshold based on word length
        adaptive_threshold = min(self.english_threshold, len(word) * 0.4)
        
        return best_distance > adaptive_threshold, correction1, correction2, best_distance
    
    def _check_hindi_word(self, word: str) -> Tuple[bool, Optional[str], Optional[str], float]:
        """Check if a Hindi word is considered an error and provide Hindi corrections"""
        if word in self.hindi_dict:
            return False, None, None, 0.0
        
        # Find best matches in Hindi dictionary
        matches = []
        
        # Use larger sample size for Hindi
        sample_size = min(5000, len(self.hindi_dict))
        sample_dict = list(self.hindi_dict)[:sample_size]
        
        for dict_word in sample_dict:
            # Skip dictionary words with very different lengths
            if abs(len(dict_word) - len(word)) > 3:
                continue
                
            # Calculate distance for Hindi
            word_distance = distance(word, dict_word)
            matches.append((dict_word, word_distance))
        
        # Sort by distance
        matches.sort(key=lambda x: x[1])
        
        # Get top two matches if available
        correction1 = matches[0][0] if matches else None
        correction2 = matches[1][0] if len(matches) > 1 else None
        
        # Get the best distance
        best_distance = matches[0][1] if matches else float('inf')
        
        # More lenient threshold for Hindi and adaptive to word length
        adaptive_threshold = min(self.hindi_threshold, len(word) * 0.5)
        
        return best_distance > adaptive_threshold, correction1, correction2, best_distance

class CSVErrorDetector:
    """Main class for CSV OCR error detection"""
    def __init__(self, english_dict_path: str, hindi_dict_path: str,
                 english_external_path: str = "english_external.txt", 
                 hindi_external_path: str = "hindi_external.txt"):
        self.dict_loader = DictionaryLoader()
        self.english_dict = self.dict_loader.load_dictionary(english_dict_path)
        self.hindi_dict = self.dict_loader.load_dictionary(hindi_dict_path)
        
        # Load external word lists
        self.english_external = self.dict_loader.load_external_words(english_external_path)
        self.hindi_external = self.dict_loader.load_external_words(hindi_external_path)
        
        # Initialize proper noun tracker
        self.proper_noun_tracker = ProperNounTracker(threshold=3)
        
        # Initialize text analyzer
        self.analyzer = TextAnalyzer(
            self.english_dict, 
            self.hindi_dict,
            self.english_external,
            self.hindi_external,
            self.proper_noun_tracker
        )
        
        # Overall statistics
        self.total_words = 0
        self.total_errors = 0
    
    def process_csv(self, input_csv_path: str, output_csv_path: str, 
                   proper_nouns_path: str = "proper_nouns.txt") -> Dict:
        """Process input CSV file and generate error analysis"""
        try:
            input_path = Path(input_csv_path)
            output_path = Path(output_csv_path)
            
            if not input_path.exists():
                raise FileNotFoundError(f"Input CSV file not found: {input_path}")
            
            # Read input CSV
            rows = []
            with open(input_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Get the header row
                rows = list(reader)
            
            # Determine overall primary language of the document
            all_text = ' '.join([' '.join(row) for row in rows])
            primary_language = self.analyzer.determine_primary_language(all_text)
            logger.info(f"Detected primary language: {primary_language}")
            
            # Analyze each row
            results = []
            for row_num, row in enumerate(rows, 1):
                row_text = ' '.join(row)
                result = self.analyzer.analyze_text(row_text, row_num, primary_language)
                results.append(result)
                
                # Update overall statistics
                self.total_words += result.total_words
                self.total_errors += result.total_errors
            
            # Write results to output CSV
            self._write_results_csv(output_path, headers, rows, results)
            
            # Save proper nouns to file
            self.proper_noun_tracker.save_to_file(proper_nouns_path)
            
            # Calculate overall error rate
            overall_error_rate = (self.total_errors / self.total_words * 100) if self.total_words > 0 else 0.0
            
            logger.info(f"Analysis completed successfully. Results written to {output_path}")
            logger.info(f"Total words: {self.total_words}, Total errors: {self.total_errors}, Overall error rate: {overall_error_rate:.2f}%")
            
            # Return statistics
            return {
                "total_rows": len(rows),
                "total_words": self.total_words,
                "total_errors": self.total_errors,
                "error_rate": overall_error_rate,
                "primary_language": primary_language
            }
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise
    
    def _write_results_csv(self, output_path: Path, headers: List[str], original_rows: List[List[str]], 
                          results: List[RowAnalysis]) -> None:
        """Write analysis results to output CSV file"""
        try:
            # Create new headers with additional columns
            new_headers = headers + [
                "accuracy_percentage", 
                "primary_language",
                "total_words", 
                "total_errors",
                "errors_details"
            ]
            
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(new_headers)
                
                for row_num, (original_row, result) in enumerate(zip(original_rows, results), 1):
                    # Calculate accuracy percentage
                    accuracy = 100.0 - result.error_percentage
                    
                    # Format error details
                    error_details = []
                    for word, (correction1, correction2, distance, _) in result.corrections.items():
                        detail = f"{word}|{distance:.2f}|{correction1 or 'N/A'}"
                        if correction2:
                            detail += f"|{correction2}"
                        error_details.append(detail)
                    
                    # Format as string with semicolon separator
                    error_details_str = "; ".join(error_details)
                    
                    # Combine the original row with the new columns
                    new_row = original_row + [
                        f"{accuracy:.2f}%",
                        result.primary_language,
                        str(result.total_words),
                        str(result.total_errors),
                        error_details_str
                    ]
                    
                    writer.writerow(new_row)
                    
            logger.info(f"Results written to CSV: {output_path}")
            
        except Exception as e:
            logger.error(f"Error writing results to CSV: {str(e)}")
            raise

def process_pdf_to_csv(pdf_path: str, output_csv_path: str) -> None:
    """Convert PDF to CSV using pytesseract OCR"""
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write header row
            writer.writerow(["text"])
            
            for i, image in enumerate(images, 1):
                # Perform OCR with support for Hindi
                text = pytesseract.image_to_string(image, lang='eng+hin')
                
                # Split text into lines
                lines = text.strip().split('\n')
                
                # Write each line as a row
                for line in lines:
                    if line.strip():  # Skip empty lines
                        writer.writerow([line.strip()])
                
        logger.info(f"PDF conversion completed. CSV written to {output_csv_path}")
        
    except Exception as e:
        logger.error(f"Error converting PDF to CSV: {str(e)}")
        raise

def main():
    # Configuration
    english_dict_path = "edic.txt"
    hindi_dict_path = "hi_IN.dic"
    english_external_path = "english_external.txt"
    hindi_external_path = "hindi_external.txt"
    input_csv_path = "17-III-07.02.2020.csv"
    output_csv_path = "output_with_analysis.csv"
    proper_nouns_path = "proper_nouns.txt"
    
    # Optional PDF processing
    pdf_mode = False
    pdf_path = "document.pdf"
    extracted_csv_path = "extracted_text.csv"
    
    try:
        if pdf_mode:
            # Step 1: Convert PDF to CSV
            process_pdf_to_csv(pdf_path, extracted_csv_path)
            input_csv_path = extracted_csv_path
        
        # Step 2: Analyze CSV for errors
        detector = CSVErrorDetector(
            english_dict_path, 
            hindi_dict_path,
            english_external_path,
            hindi_external_path
        )
        stats = detector.process_csv(input_csv_path, output_csv_path, proper_nouns_path)
        
        # Print summary statistics
        print("\n===== Analysis Summary =====")
        print(f"Primary Language: {stats['primary_language']}")
        print(f"Total Rows: {stats['total_rows']}")
        print(f"Total Words: {stats['total_words']}")
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Overall Error Rate: {stats['error_rate']:.2f}%")
        print(f"Overall Accuracy: {100 - stats['error_rate']:.2f}%")
        print("===========================\n")
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
