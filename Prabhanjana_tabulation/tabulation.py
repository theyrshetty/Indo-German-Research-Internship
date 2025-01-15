import re
import csv
import pandas as pd
import os
from typing import List, Tuple, Dict

class ParliamentProcessor:
    def __init__(self):
        self.current_page = ""
        self.current_metadata = ""
        self.current_speaker = ""
        self.current_speech = []
        self.records = []
        self.processing_started = False
        self.probable_speaker_lines = []
        self.non_speaker_words = 0
        
    def is_page_marker(self, line: str) -> bool:
        """Check if the line indicates a new page."""
        return line.strip().startswith("Page")
    
    def extract_page_number(self, line: str) -> str:
        """Extract just the number from a page marker line."""
        # Match digit sequence after "Page" and optional space/colon
        match = re.search(r'Page[\s:]*(\d+)', line)
        if match:
            return match.group(1)
        return ""
    
    def is_bold_text(self, text: str) -> bool:
        """Check if text is in bold (surrounded by **)."""
        return text.startswith("**") and text.endswith("**")
    
    def is_capitalized(self, text: str) -> bool:
        """Check if text is in capitals."""
        return text.isupper() and len(text) > 1
    
    def is_speaker_formatted(self, word: str) -> bool:
        """Check if a word has speaker formatting (bold or caps)."""
        return self.is_bold_text(word) or self.is_capitalized(word)
    
    def clean_speaker_text(self, text: str) -> str:
        """Remove ** markers from speaker text."""
        words = text.strip().split()
        cleaned_words = []
        for word in words:
            if self.is_bold_text(word):
                cleaned_words.append(word[2:-2])  # Remove ** from start and end
            else:
                cleaned_words.append(word)
        return ' '.join(cleaned_words)
    
    def validate_speaker_line(self, text: str) -> bool:
        """Validate if a line is a legitimate speaker line based on formatting."""
        # Split on colon or semicolon if present
        separator = ':' if ':' in text else ';' if ';' in text else None
        if not separator:
            return False
            
        speaker_part = text.split(separator)[0].strip()
        words = speaker_part.split()
        
        # Require at least one word to be properly formatted
        if not any(self.is_speaker_formatted(word) for word in words):
            return False
            
        # Count non-speaker formatted words
        non_speaker_words = sum(1 for word in words if not self.is_speaker_formatted(word))
        
        # Allow up to 3 non-speaker formatted words
        return non_speaker_words <= 3
    
    def is_probable_speaker(self, text: str) -> bool:
        """Check if line might be a speaker based on first two words."""
        words = text.strip().split()
        if len(words) < 1:
            return False
            
        # Check first two words
        check_words = words[:2] if len(words) >= 2 else words[:1]
        return any(self.is_speaker_formatted(word) for word in check_words)
    
    def count_non_speaker_words(self, line: str) -> int:
        """Count words that don't have speaker formatting in a line."""
        words = line.strip().split()
        return sum(1 for word in words if not self.is_speaker_formatted(word))
    
    def process_line(self, line: str):
        """Process each line of the text file."""
        line = line.strip()
        if not line:
            return
            
        # Check for page marker and extract just the number
        if self.is_page_marker(line):
            self.current_page = self.extract_page_number(line)
            return
            
        # Check for metadata (date line)
        if re.match(r'\d{2}\.\d{2}\.\d{4}', line):
            self.current_metadata = line
            return
            
        # Check for session start marker
        if "[HON. SPEAKER in the Chair]" in line:
            self.processing_started = True
            return
            
        if not self.processing_started:
            return

        # If we're collecting probable speaker lines
        if self.probable_speaker_lines:
            if ':' in line or ';' in line:
                # Add this line to buffer and validate the combined text
                self.probable_speaker_lines.append(line)
                full_text = ' '.join(self.probable_speaker_lines)
                
                if self.validate_speaker_line(full_text):
                    if self.current_speaker:
                        self.save_current_record()
                    
                    separator = ':' if ':' in full_text else ';'
                    speaker_part = full_text.split(separator)[0].strip()
                    self.current_speaker = self.clean_speaker_text(speaker_part)
                    speech_start = full_text[full_text.find(separator)+1:].strip()
                    self.current_speech = [speech_start] if speech_start else []
                else:
                    # If validation fails, treat as speech
                    if self.current_speaker:
                        self.current_speech.extend(self.probable_speaker_lines)
                
                self.probable_speaker_lines = []
                self.non_speaker_words = 0
            else:
                # Count non-speaker formatted words
                non_speaker = self.count_non_speaker_words(line)
                self.non_speaker_words += non_speaker
                
                # If too many non-speaker words, discard the buffer and treat as speech
                if self.non_speaker_words > 3:
                    if self.current_speaker:
                        self.current_speech.extend(self.probable_speaker_lines)
                        self.current_speech.append(line)
                    self.probable_speaker_lines = []
                    self.non_speaker_words = 0
                else:
                    # Continue collecting probable speaker lines
                    self.probable_speaker_lines.append(line)
            return
            
        # Check for direct speaker line with proper formatting
        if (':' in line or ';' in line) and self.validate_speaker_line(line):
            if self.current_speaker:
                self.save_current_record()
            separator = ':' if ':' in line else ';'
            speaker_part = line.split(separator)[0].strip()
            self.current_speaker = self.clean_speaker_text(speaker_part)
            speech_start = line[line.find(separator)+1:].strip()
            self.current_speech = [speech_start] if speech_start else []
            return
            
        # Start new probable speaker collection if first two words match
        if self.is_probable_speaker(line):
            self.probable_speaker_lines = [line]
            self.non_speaker_words = self.count_non_speaker_words(line)
        else:
            # Regular speech line
            if self.current_speaker:
                self.current_speech.append(line)
    
    def save_current_record(self):
        """Save the current speaker and speech as a record."""
        if self.current_speaker and self.current_speech:
            self.records.append({
                'page': self.current_page,
                'metadata': self.current_metadata,
                'speaker': self.current_speaker,
                'speech': ' '.join(self.current_speech).strip()
            })
    
    def process_file(self, input_file: str):
        """Process the input file and generate CSV output with automatic filename."""
        # Generate output filename by replacing .txt with .csv
        output_file = os.path.splitext(input_file)[0] + '.csv'
        
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                self.process_line(line)
            
            # Process any remaining buffer
            if self.probable_speaker_lines:
                if self.non_speaker_words > 3:
                    # If too many non-speaker words, treat as speech
                    if self.current_speaker:
                        self.current_speech.extend(self.probable_speaker_lines)
                else:
                    # If reasonable number of non-speaker words, treat as new speaker
                    if self.current_speaker:
                        self.save_current_record()
                    combined_text = ' '.join(self.probable_speaker_lines)
                    self.current_speaker = self.clean_speaker_text(combined_text)
                    self.current_speech = []
            
            # Save last record if exists
            if self.current_speaker:
                self.save_current_record()
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(self.records)
        df.to_csv(output_file, index=False, encoding='utf-8')

def main():
    processor = ParliamentProcessor()
    processor.process_file('16-III-01.12.2014.txt')

if __name__ == "__main__":
    main()