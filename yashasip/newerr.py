import os
import re
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
import logging
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from Levenshtein import distance
from spellchecker import SpellChecker


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WordPosition(NamedTuple):
    """Store position information for a word"""
    row: int
    column: str
    word: int
    text: str  # The context text

@dataclass
class ErrorDetail:
    """Detailed information about a specific error"""
    original_word: str
    error_distance: float
    suggested_corrections: List[Tuple[str, float]]  # List of (correction, distance) tuples
    position: WordPosition

@dataclass
class RowAnalysis:
    """Data class to store analysis results for a single row"""
    row_number: int
    column_stats: Dict[str, Dict[str, float]]  # Maps column name to stats
    english_corrections: Dict[str, Tuple[str, WordPosition]]  # Maps incorrect words to (correction, position)
    hindi_corrections: Dict[str, Tuple[str, WordPosition]]  # Maps incorrect words to (correction, position)
    error_details: List[ErrorDetail]  # Detailed error information
    accuracy_percentage: float
    primary_language: str
    total_errors: int

class DictionaryLoader:
    """Handles dictionary loading and caching"""
    @staticmethod
    def load_dictionary(file_path: str) -> Set[str]:
        """Load dictionary from file and return as set of words"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return {word.strip().lower() for word in f if word.strip()}
        except Exception as e:
            logger.error(f"Error loading dictionary {file_path}: {str(e)}")
            # If dictionary file is not available, use a minimal set as fallback
            logger.warning(f"Using fallback minimal dictionary")
            if "hi_IN" in file_path or "hindi" in file_path.lower():
                # Minimal Hindi dictionary for fallback
                return {"में", "है", "का", "की", "के", "एक", "से", "हैं", "को", "पर", "इस", "होता", "कि", "जो", "ने"}
            else:
                # Minimal English dictionary for fallback
                return {"the", "a", "an", "in", "on", "of", "and", "to", "is", "are", "was", "were", "be", "been", "have", "has", "had"}

class TextAnalyzer:
    """Handles text analysis and error detection"""
    def __init__(self, english_dict: Set[str], hindi_dict: Set[str]):
        self.english_spellchecker = SpellChecker()
        self.english_dict = set(self.english_spellchecker.word_frequency.words)
        self.hindi_dict = hindi_dict
        # Different thresholds for English and Hindi
        self.english_threshold = 2
        self.hindi_threshold = 3.5  # More lenient threshold for Hindi
    
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
    
    def find_best_corrections(self, word: str, dictionary: Set[str], max_corrections: int = 3) -> List[Tuple[str, float]]:
        """Find best corrections for a word with normalized distance scores"""
        corrections = []
        word_len = len(word)
        
        # Use larger sample for better corrections
        sample_size = min(5000, len(dictionary))
        sample_dict = list(dictionary)[:sample_size]
        
        # For short words, check the entire dictionary
        if word_len <= 3:
            sample_dict = dictionary
        
        for dict_word in sample_dict:
            # Skip dictionary words with very different lengths
            if abs(len(dict_word) - word_len) > max(2, word_len * 0.5):
                continue
            
            # Calculate normalized Levenshtein distance
            edit_distance = distance(word, dict_word)
            normalized_distance = edit_distance / max(word_len, len(dict_word))
            
            corrections.append((dict_word, normalized_distance))
        
        # Sort by distance and return top corrections
        corrections.sort(key=lambda x: x[1])
        return corrections[:max_corrections]
    
    def calculate_cell_accuracy(self, text: str, row_num: int, column_name: str) -> Tuple[int, int, float, Dict[str, Tuple[str, WordPosition]], int, int, float, Dict[str, Tuple[str, WordPosition]], List[ErrorDetail]]:
        """Calculate error percentage for both English and Hindi text with position tracking and detailed error info"""
        if pd.isna(text) or text == '':
            return 0, 0, 0.0, {}, 0, 0, 0.0, {}, []
        
        text = str(text)  # Ensure text is a string
        
        # Improved word extraction - handle punctuation and special characters better
        words = re.findall(r'\b[\w\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+\b', text)
        
        eng_total = 0
        hin_total = 0
        eng_errors = 0
        hin_errors = 0
        eng_corrections = {}
        hin_corrections = {}
        error_details = []
        
        # Process words in this cell
        for word_num, word in enumerate(words, 1):
            # Skip very short words (likely not meaningful)
            if len(word) <= 1:
                continue
                
            # Get context (the whole cell content serves as context for CSV)
            context = text[:100] + "..." if len(text) > 100 else text
            position = WordPosition(row=row_num, column=column_name, word=word_num, text=context)
            
            if self.is_english_word(word):
                eng_total += 1
                is_error, correction, error_distance, all_corrections = self._check_word_detailed(word.lower(), self.english_dict, self.english_threshold)
                if is_error:
                    eng_errors += 1
                    if correction:
                        eng_corrections[word.lower()] = (correction, position)
                    
                    # Create detailed error information
                    error_detail = ErrorDetail(
                        original_word=word,
                        error_distance=error_distance,
                        suggested_corrections=all_corrections,
                        position=position
                    )
                    error_details.append(error_detail)
            
            elif self.is_hindi_word(word):
                hin_total += 1
                is_error, correction, error_distance, all_corrections = self._check_word_detailed(word, self.hindi_dict, self.hindi_threshold)
                if is_error:
                    hin_errors += 1
                    if correction:
                        hin_corrections[word] = (correction, position)
                    
                    # Create detailed error information
                    error_detail = ErrorDetail(
                        original_word=word,
                        error_distance=error_distance,
                        suggested_corrections=all_corrections,
                        position=position
                    )
                    error_details.append(error_detail)
            
            else:
                # Try to classify mixed words based on character majority
                hindi_chars = sum(1 for c in word if '\u0900' <= c <= '\u097F' or '\u1CD0' <= c <= '\u1CFF' or '\uA8E0' <= c <= '\uA8FF')
                english_chars = sum(1 for c in word if 'a' <= c.lower() <= 'z')
                
                if hindi_chars > english_chars:
                    hin_total += 1
                    is_error, correction, error_distance, all_corrections = self._check_word_detailed(word, self.hindi_dict, self.hindi_threshold)
                    if is_error:
                        hin_errors += 1
                        if correction:
                            hin_corrections[word] = (correction, position)
                        
                        error_detail = ErrorDetail(
                            original_word=word,
                            error_distance=error_distance,
                            suggested_corrections=all_corrections,
                            position=position
                        )
                        error_details.append(error_detail)
                elif english_chars > hindi_chars:
                    eng_total += 1
                    is_error, correction, error_distance, all_corrections = self._check_word_detailed(word.lower(), self.english_dict, self.english_threshold)
                    if is_error:
                        eng_errors += 1
                        if correction:
                            eng_corrections[word.lower()] = (correction, position)
                        
                        error_detail = ErrorDetail(
                            original_word=word,
                            error_distance=error_distance,
                            suggested_corrections=all_corrections,
                            position=position
                        )
                        error_details.append(error_detail)
        
        # Calculate error percentages
        eng_error_percentage = (eng_errors / eng_total * 100) if eng_total > 0 else 0.0
        hin_error_percentage = (hin_errors / hin_total * 100) if hin_total > 0 else 0.0
        
        # Calculate accuracy percentages (100 - error percentage)
        eng_accuracy = 100 - eng_error_percentage
        hin_accuracy = 100 - hin_error_percentage
        
        return eng_total, eng_errors, eng_accuracy, eng_corrections, hin_total, hin_errors, hin_accuracy, hin_corrections, error_details
    
    def _check_word_detailed(self, word: str, dictionary: Set[str], threshold: float) -> Tuple[bool, Optional[str], float, List[Tuple[str, float]]]:
        """Check if a word is an error and return detailed correction information"""
        # Skip very short words
        if len(word) <= 2:
            return False, None, 0.0, []
            
        # If the word is already in the dictionary, it's correct
        if dictionary == self.english_dict:
            if word in self.english_spellchecker:
                return False, None, 0.0, []
            else:
                suggestions = list(self.english_spellchecker.candidates(word))
                suggestions = sorted(suggestions, key=lambda s: distance(word, s))
                suggestions_with_distance = [(s, distance(word, s) / max(len(word), len(s))) for s in suggestions]
                best = suggestions_with_distance[:3]
                best_correction = best[0][0] if best else None
                best_distance = best[0][1] if best else 1.0
                is_error = best_distance > 0.2  # you can tweak threshold
                return is_error, best_correction, best_distance, best
        
        # Check for capitalized versions (proper nouns)
        if word.capitalize() in dictionary or word.upper() in dictionary or word.lower() in dictionary:
            return False, None, 0.0, []
        
        # Skip numbers, alphanumeric combinations, and words with special characters
        if any(c.isdigit() for c in word) or not word.isalpha():
            return False, None, 0.0, []
        
        # Find best corrections
        corrections = self.find_best_corrections(word, dictionary)
        
        if not corrections:
            # If no corrections found, mark as correct for now
            # This addresses cases where the dictionary is incomplete
            return False, None, 1.0, []
        
        best_correction, best_distance = corrections[0]
        
        # Use more lenient thresholds based on word length
        word_len = len(word)
        if word_len <= 4:
            # Be more lenient with short words
            adaptive_threshold = threshold * 0.7
        elif word_len <= 6:
            adaptive_threshold = threshold * 0.5
        else:
            adaptive_threshold = threshold * 0.4
        
        # Apply different thresholds based on language
        if dictionary == self.hindi_dict:
            # Even more lenient for Hindi
            adaptive_threshold = min(0.8, adaptive_threshold * 1.5)
        
        if best_distance <= adaptive_threshold:
            return False, None, best_distance, corrections
        else:
            return True, best_correction, best_distance, corrections

class CSVErrorDetector:
    """Main class for CSV OCR error detection"""
    def __init__(self, english_dict_path: str = "edic.txt", hindi_dict_path: str = "hi_IN.dic"):
        self.dict_loader = DictionaryLoader()
        try:
            self.english_dict = self.dict_loader.load_dictionary(english_dict_path)
            self.hindi_dict = self.dict_loader.load_dictionary(hindi_dict_path)
        except:
            logger.warning("Dictionary files not found. Using minimal fallback dictionaries.")
            self.english_dict = self.dict_loader.load_dictionary("")  # Will use fallback
            self.hindi_dict = self.dict_loader.load_dictionary("")   # Will use fallback
            
        self.analyzer = TextAnalyzer(self.english_dict, self.hindi_dict)
        # Track all errors across rows
        self.all_english_errors = {}
        self.all_hindi_errors = {}
    
    def determine_primary_language(self, text: str) -> str:
        """Determine the primary language of the text"""
        if pd.isna(text) or text == '':
            return "Unknown"
        
        text = str(text)
        words = re.findall(r'\b[\w\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+\b', text)
        
        english_count = 0
        hindi_count = 0
        
        for word in words:
            if self.analyzer.is_english_word(word):
                english_count += 1
            elif self.analyzer.is_hindi_word(word):
                hindi_count += 1
        
        if english_count > hindi_count:
            return "English"
        elif hindi_count > english_count:
            return "Hindi"
        else:
            return "Mixed" if english_count > 0 or hindi_count > 0 else "Unknown"
    
    def process_csv(self, input_csv_path: str, output_csv_path: str, output_report_path: str) -> None:
        """Process input CSV file and generate enhanced output with error analysis"""
        try:
            if not os.path.exists(input_csv_path):
                raise FileNotFoundError(f"Input file not found: {input_csv_path}")
            
            # Load CSV data
            df = pd.read_csv(input_csv_path)
            
            # Analyze CSV data
            analysis_results = self._analyze_csv(df)
            
            # Generate overall CSV statistics
            csv_stats = self._calculate_csv_stats(analysis_results)
            
            # Create enhanced DataFrame with additional columns
            enhanced_df = self._create_enhanced_dataframe(df, analysis_results)
            
            # Save enhanced CSV
            enhanced_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            
            # Write analysis report
            self._write_results(output_report_path, df, analysis_results, csv_stats)
            
            logger.info(f"Analysis completed successfully. Enhanced CSV: {output_csv_path}, Report: {output_report_path}")
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise
    
    def _create_enhanced_dataframe(self, df: pd.DataFrame, results: List[RowAnalysis]) -> pd.DataFrame:
        """Create enhanced DataFrame with additional accuracy and error columns"""
        # Start with original DataFrame
        enhanced_df = df.copy()
        
        # Add new columns
        accuracy_percentages = []
        primary_languages = []
        error_counts = []
        error_details_cols = []
        
        for result in results:
            accuracy_percentages.append(round(result.accuracy_percentage, 2))
            primary_languages.append(result.primary_language)
            error_counts.append(result.total_errors)
            
            # Format error details for CSV
            error_details_str = self._format_error_details_for_csv(result.error_details)
            error_details_cols.append(error_details_str)
        
        # Add the new columns to DataFrame
        enhanced_df['Accuracy_Percentage'] = accuracy_percentages
        enhanced_df['Primary_Language'] = primary_languages
        enhanced_df['Number_of_Errors'] = error_counts
        enhanced_df['Error_Details'] = error_details_cols
        
        return enhanced_df
    
    def _format_error_details_for_csv(self, error_details: List[ErrorDetail]) -> str:
        """Format error details as a string for CSV storage"""
        if not error_details:
            return ""
        
        formatted_errors = []
        for error in error_details:
            # Get first two suggestions
            suggestions = error.suggested_corrections[:2]
            if len(suggestions) >= 2:
                first_sugg, first_dist = suggestions[0]
                second_sugg, second_dist = suggestions[1]
                error_str = f"[{error.original_word}|{error.error_distance:.3f}|{first_sugg}|{second_sugg}]"
            elif len(suggestions) == 1:
                first_sugg, first_dist = suggestions[0]
                error_str = f"[{error.original_word}|{error.error_distance:.3f}|{first_sugg}|N/A]"
            else:
                error_str = f"[{error.original_word}|{error.error_distance:.3f}|N/A|N/A]"
            
            formatted_errors.append(error_str)
        
        return "; ".join(formatted_errors)
    
    def _analyze_csv(self, df: pd.DataFrame) -> List[RowAnalysis]:
        """Analyze each row of the CSV and return results"""
        results = []
        
        # Analyze each row
        for row_idx, row in df.iterrows():
            row_num = row_idx + 2  # Account for 1-based indexing and header row
            column_stats = {}
            all_eng_corrections = {}
            all_hin_corrections = {}
            all_error_details = []
            
            # Combine all text in the row to determine primary language
            row_text = " ".join([str(cell) for cell in row.values if pd.notna(cell)])
            primary_language = self.determine_primary_language(row_text)
            
            total_words_in_row = 0
            total_errors_in_row = 0
            
            # Analyze each column in the row
            for col_name in df.columns:
                cell_value = row[col_name]
                
                # Skip analysis if cell is empty
                if pd.isna(cell_value) or cell_value == '':
                    column_stats[col_name] = {
                        'english_total': 0,
                        'english_errors': 0,
                        'english_accuracy': 100.0,
                        'hindi_total': 0,
                        'hindi_errors': 0,
                        'hindi_accuracy': 100.0,
                        'total_words': 0,
                        'overall_accuracy': 100.0
                    }
                    continue
                
                # Analyze cell content
                eng_total, eng_errors, eng_accuracy, eng_corrections, \
                hin_total, hin_errors, hin_accuracy, hin_corrections, error_details = self.analyzer.calculate_cell_accuracy(
                    str(cell_value), row_num, col_name
                )
                
                # Update global error dictionaries
                for word, (correction, _) in eng_corrections.items():
                    self.all_english_errors[word] = correction
                    all_eng_corrections[word] = (correction, WordPosition(
                        row=row_num, column=col_name, word=0, text=str(cell_value)[:100]
                    ))
                
                for word, (correction, _) in hin_corrections.items():
                    self.all_hindi_errors[word] = correction
                    all_hin_corrections[word] = (correction, WordPosition(
                        row=row_num, column=col_name, word=0, text=str(cell_value)[:100]
                    ))
                
                # Collect error details
                all_error_details.extend(error_details)
                
                # Calculate overall cell accuracy
                total_words = eng_total + hin_total
                total_errors = eng_errors + hin_errors
                overall_accuracy = 100 - (total_errors / total_words * 100) if total_words > 0 else 100.0
                
                # Update row totals
                total_words_in_row += total_words
                total_errors_in_row += total_errors
                
                # Store column stats
                column_stats[col_name] = {
                    'english_total': eng_total,
                    'english_errors': eng_errors,
                    'english_accuracy': eng_accuracy,
                    'hindi_total': hin_total,
                    'hindi_errors': hin_errors,
                    'hindi_accuracy': hin_accuracy,
                    'total_words': total_words,
                    'overall_accuracy': overall_accuracy
                }
            
            # Calculate row-wise accuracy
            row_accuracy = 100 - (total_errors_in_row / total_words_in_row * 100) if total_words_in_row > 0 else 100.0
            
            # Create row analysis object
            row_analysis = RowAnalysis(
                row_number=row_num,
                column_stats=column_stats,
                english_corrections=all_eng_corrections,
                hindi_corrections=all_hin_corrections,
                error_details=all_error_details,
                accuracy_percentage=row_accuracy,
                primary_language=primary_language,
                total_errors=total_errors_in_row
            )
            
            results.append(row_analysis)
        
        return results
    
    def _calculate_csv_stats(self, results: List[RowAnalysis]) -> Dict:
        """Calculate overall statistics for the CSV"""
        # Initialize counters
        total_english_words = 0
        total_english_errors = 0
        total_hindi_words = 0
        total_hindi_errors = 0
        
        # For column-wise statistics
        column_stats = defaultdict(lambda: {
            'english_total': 0,
            'english_errors': 0,
            'hindi_total': 0,
            'hindi_errors': 0,
            'total_words': 0,
            'total_errors': 0
        })
        
        # Aggregate statistics
        for result in results:
            for col_name, stats in result.column_stats.items():
                total_english_words += stats['english_total']
                total_english_errors += stats['english_errors']
                total_hindi_words += stats['hindi_total']
                total_hindi_errors += stats['hindi_errors']
                
                # Update column-specific stats
                column_stats[col_name]['english_total'] += stats['english_total']
                column_stats[col_name]['english_errors'] += stats['english_errors']
                column_stats[col_name]['hindi_total'] += stats['hindi_total']
                column_stats[col_name]['hindi_errors'] += stats['hindi_errors']
                column_stats[col_name]['total_words'] += stats['total_words']
                column_stats[col_name]['total_errors'] += stats['english_errors'] + stats['hindi_errors']
        
        # Calculate overall accuracy percentages
        overall_english_accuracy = 100 - (total_english_errors / total_english_words * 100) if total_english_words > 0 else 100.0
        overall_hindi_accuracy = 100 - (total_hindi_errors / total_hindi_words * 100) if total_hindi_words > 0 else 100.0
        
        total_words = total_english_words + total_hindi_words
        total_errors = total_english_errors + total_hindi_errors
        overall_accuracy = 100 - (total_errors / total_words * 100) if total_words > 0 else 100.0
        
        # Calculate column-wise accuracy
        for col_name in column_stats:
            col = column_stats[col_name]
            col['english_accuracy'] = 100 - (col['english_errors'] / col['english_total'] * 100) if col['english_total'] > 0 else 100.0
            col['hindi_accuracy'] = 100 - (col['hindi_errors'] / col['hindi_total'] * 100) if col['hindi_total'] > 0 else 100.0
            col['overall_accuracy'] = 100 - (col['total_errors'] / col['total_words'] * 100) if col['total_words'] > 0 else 100.0
        
        return {
            'total_english_words': total_english_words,
            'total_english_errors': total_english_errors,
            'overall_english_accuracy': overall_english_accuracy,
            'total_hindi_words': total_hindi_words,
            'total_hindi_errors': total_hindi_errors,
            'overall_hindi_accuracy': overall_hindi_accuracy,
            'total_words': total_words,
            'total_errors': total_errors,
            'overall_accuracy': overall_accuracy,
            'column_stats': column_stats
        }
    
    def _write_results(self, output_path: str, df: pd.DataFrame, results: List[RowAnalysis], csv_stats: Dict) -> None:
        """Write analysis results to output file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write CSV summary
            f.write("=== ENHANCED CSV OCR ACCURACY ANALYSIS ===\n\n")
            
            f.write("--- OVERALL CSV ACCURACY ---\n")
            f.write(f"Total Words: {csv_stats['total_words']}\n")
            f.write(f"Total Errors: {csv_stats['total_errors']}\n")
            f.write(f"Overall Accuracy: {csv_stats['overall_accuracy']:.2f}%\n\n")
            
            f.write(f"English Words: {csv_stats['total_english_words']}\n")
            f.write(f"English Errors: {csv_stats['total_english_errors']}\n") 
            f.write(f"English Accuracy: {csv_stats['overall_english_accuracy']:.2f}%\n\n")
            
            f.write(f"Hindi Words: {csv_stats['total_hindi_words']}\n")
            f.write(f"Hindi Errors: {csv_stats['total_hindi_errors']}\n")
            f.write(f"Hindi Accuracy: {csv_stats['overall_hindi_accuracy']:.2f}%\n\n")
            
            # Write column-wise accuracy
            f.write("--- COLUMN-WISE ACCURACY ---\n")
            f.write("Column | Total Words | Overall Accuracy | English Words | English Accuracy | Hindi Words | Hindi Accuracy\n")
            f.write("-" * 120 + "\n")
            
            for col_name, stats in csv_stats['column_stats'].items():
                f.write(
                    f"{col_name} | {stats['total_words']} | {stats['overall_accuracy']:.2f}% | "
                    f"{stats['english_total']} | {stats['english_accuracy']:.2f}% | "
                    f"{stats['hindi_total']} | {stats['hindi_accuracy']:.2f}%\n"
                )
            
            # Write row-wise accuracy with enhanced details
            f.write("\n\n--- ROW-WISE ACCURACY WITH DETAILED METRICS ---\n")
            f.write("Row | Accuracy % | Primary Lang | Errors | Total Words | Error Rate\n")
            f.write("-" * 80 + "\n")
            
            for result in results:
                # Calculate row-wise totals
                row_eng_total = sum(stats['english_total'] for stats in result.column_stats.values())
                row_hin_total = sum(stats['hindi_total'] for stats in result.column_stats.values())
                row_total_words = row_eng_total + row_hin_total
                
                error_rate = (result.total_errors / row_total_words * 100) if row_total_words > 0 else 0.0
                
                f.write(
                    f"{result.row_number} | {result.accuracy_percentage:.2f}% | "
                    f"{result.primary_language} | {result.total_errors} | "
                    f"{row_total_words} | {error_rate:.2f}%\n"
                )
            
            # Detailed error report for each row with distance metrics
            f.write("\n\n--- DETAILED ERROR REPORT WITH DISTANCE METRICS ---\n")
            
            for row_idx, result in enumerate(results):
                row_num = result.row_number
                
                if result.error_details:
                    f.write(f"\nRow {row_num} Error Details (Accuracy: {result.accuracy_percentage:.2f}%):\n")
                    f.write("  Original Word | Error Distance | First Correction | Second Correction | Column | Context\n")
                    f.write("  " + "-" * 120 + "\n")
                    
                    for error in result.error_details:
                        suggestions = error.suggested_corrections
                        first_corr = suggestions[0][0] if len(suggestions) > 0 else "N/A"
                        second_corr = suggestions[1][0] if len(suggestions) > 1 else "N/A"
                        
                        f.write(
                            f"  {error.original_word} | {error.error_distance:.3f} | "
                            f"{first_corr} | {second_corr} | "
                            f"{error.position.column} | {error.position.text[:30]}...\n"
                        )
            
            # Write complete list of all errors with suggestions
            f.write("\n\n--- COMPLETE ERROR DICTIONARY WITH MULTIPLE SUGGESTIONS ---\n")
            
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
            
            # Write CSV enhancement information
            f.write("\n\n--- CSV ENHANCEMENT DETAILS ---\n")
            f.write("The enhanced CSV includes the following additional columns:\n")
            f.write("1. Accuracy_Percentage: Overall accuracy for each row\n")
            f.write("2. Primary_Language: Detected primary language (English/Hindi/Mixed/Unknown)\n")
            f.write("3. Number_of_Errors: Total number of errors found in the row\n")
            f.write("4. Error_Details: Formatted string with error details in format:\n")
            f.write("   [original_word|error_distance|first_suggestion|second_suggestion]\n")
            f.write("   Multiple errors are separated by semicolons\n")
            f.write("   Error distance ranges from 0.0 (perfect match) to 1.0 (completely different)\n")

def main():
    # Configuration
    english_dict_path = "edic.txt"
    hindi_dict_path = "hi_IN.dic"
    input_path = "sample_parliament_data.csv"
    output_csv_path = "enhanced_parliament_data.csv"
    output_report_path = "detailed_csv_analysis_report.txt"
    
    try:
        # Analyze CSV for errors and create enhanced output
        detector = CSVErrorDetector(english_dict_path, hindi_dict_path)
        detector.process_csv(input_path, output_csv_path, output_report_path)
        print(f"Analysis complete.")
        print(f"Enhanced CSV saved to: {output_csv_path}")
        print(f"Detailed report saved to: {output_report_path}")
        
        # Print summary statistics
        print("\n--- SUMMARY ---")
        print("The enhanced CSV now includes:")
        print("• Accuracy_Percentage: Spelling accuracy for each row")
        print("• Primary_Language: Detected language (English/Hindi/Mixed)")
        print("• Number_of_Errors: Count of spelling errors")
        print("• Error_Details: Detailed error information with distance metrics")
        print("  Format: [original_word|distance|first_correction|second_correction]")
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
