import pandas as pd
import re
import string
from difflib import SequenceMatcher
import numpy as np
import os
import sys

# Function to normalize names for better comparison
def normalize_name(name):
    if not isinstance(name, str):
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove punctuation
    name = name.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

# Function to check if a speaker name is the Speaker/Chair of the House
def is_speaker_chair(speaker_name):
    # Handle None and non-string values
    if not isinstance(speaker_name, str) or not speaker_name:
        return False
    
    # Normalize the speaker name
    norm_name = normalize_name(speaker_name)
    
    # Additional check to make sure norm_name is not empty
    if not norm_name:
        return False
    
    # List of variations of Speaker/Chair references in English and Hindi
    speaker_variations = [
        'hon speaker', 'honorable speaker', 'माननीय अध्यक्ष'
    ]
    
    # Check if the normalized name contains any of the speaker variations
    for variation in speaker_variations:
        if variation in norm_name or norm_name in variation:
            return True
    
    return False

def is_chair_chair(speaker_name):
    # Handle None and non-string values
    if not isinstance(speaker_name, str) or not speaker_name:
        return False
    
    # Normalize the speaker name
    norm_name = normalize_name(speaker_name)
    
    # Additional check to make sure norm_name is not empty
    if not norm_name:
        return False
    
    # List of variations of Speaker/Chair references in English and Hindi
    speaker_variations = [
        'hon chairperson', 'माननीय सभापति'
    ]
    
    # Check if the normalized name contains any of the speaker variations
    for variation in speaker_variations:
        if variation in norm_name or norm_name in variation:
            return True
    
    return False

# Function to handle Hindi name variations with and without spaces
def normalize_hindi_name(name):
    if not isinstance(name, str) or not name:
        return "", ""
    
    # Regular normalization
    norm_name = normalize_name(name)
    
    # Create a version with spaces removed
    no_space_name = norm_name.replace(" ", "")
    
    return norm_name, no_space_name

# Function to check if a speaker name contains words from an MP name
def check_name_words_match(speaker_name, mp_name, is_hindi=False):
    if not isinstance(speaker_name, str) or not isinstance(mp_name, str):
        return 0, []
    
    # For Hindi names, create both standard and no-space versions
    if is_hindi:
        norm_speaker, speaker_no_space = normalize_hindi_name(speaker_name)
        norm_mp, mp_no_space = normalize_hindi_name(mp_name)
        
        # Check for full name match first (with or without spaces)
        if norm_speaker == norm_mp or speaker_no_space == mp_no_space:
            return 1.0, norm_mp.split()  # Perfect match
        
        # Check for combined words match (handle cases like "हर्ष वर्धन" vs "हर्षवर्धन")
        if speaker_no_space == mp_no_space:
            return 0.95, norm_mp.split()  # Almost perfect match
        
        # Try to match no-space version of speaker to mp with spaces and vice versa
        if speaker_no_space == norm_mp or norm_speaker == mp_no_space:
            return 0.90, norm_mp.split()  # Strong match with space variation
            
        # Add substring matching within no-space versions 
        # (handles cases like "डॉ. हर्ष वर्धन" vs "हर्षवर्धन")
        if mp_no_space in speaker_no_space:
            # Calculate how significant the match is based on relative lengths
            match_ratio = len(mp_no_space) / len(speaker_no_space)
            # If the MP name makes up at least 70% of the speaker name (excluding prefixes)
            if match_ratio >= 0.7:
                return 0.85, norm_mp.split()  # Strong match with prefix
        # Also check the reverse (if speaker name is contained in MP name)
        elif speaker_no_space in mp_no_space:
            match_ratio = len(speaker_no_space) / len(mp_no_space)
            if match_ratio >= 0.7:
                return 0.80, norm_mp.split()  # Good match with speaker being subset
    else:
        # Regular English name processing
        norm_speaker = normalize_name(speaker_name)
        norm_mp = normalize_name(mp_name)
        
        # Check for full name match
        if norm_speaker == norm_mp:
            return 1.0, norm_mp.split()  # Perfect match
    
    # If no full match (or no-space match for Hindi), proceed with word-by-word matching
    if not norm_speaker or not norm_mp:
        return 0, []
    
    # Split names into words
    speaker_words = set(norm_speaker.split())
    mp_words = norm_mp.split()
    
    # Check for each word in the MP name
    matched_words = []
    for word in mp_words:
        if word in speaker_words:
            matched_words.append(word)
    
    # Calculate match score
    if matched_words:
        # Prioritize matches that have more unique words
        match_score = len(matched_words) / len(mp_words)
        
        # Add a small boost for sequential matches
        mp_name_str = ' '.join(mp_words)
        for i in range(len(matched_words) - 1):
            if mp_name_str.find(f"{matched_words[i]} {matched_words[i+1]}") >= 0:
                match_score += 0.05  # Small boost for sequential matches
        
        return match_score, matched_words
    
    return 0, []

# Calculate string similarity as a backup/additional metric
def calculate_string_similarity(str1, str2, is_hindi=False):
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0
    
    if is_hindi:
        # For Hindi strings, compare both with and without spaces
        norm_str1, no_space_str1 = normalize_hindi_name(str1)
        norm_str2, no_space_str2 = normalize_hindi_name(str2)
        
        # Calculate similarity for both versions
        regular_sim = SequenceMatcher(None, norm_str1, norm_str2).ratio() if norm_str1 and norm_str2 else 0
        no_space_sim = SequenceMatcher(None, no_space_str1, no_space_str2).ratio() if no_space_str1 and no_space_str2 else 0
        
        # Also calculate substring similarity scores
        substring_sim = 0
        if no_space_str2 in no_space_str1:
            # Calculate match quality based on length ratio
            match_ratio = len(no_space_str2) / len(no_space_str1)
            substring_sim = 0.7 + (0.3 * match_ratio)  # Score between 0.7-1.0 based on coverage
        elif no_space_str1 in no_space_str2:
            match_ratio = len(no_space_str1) / len(no_space_str2)
            substring_sim = 0.7 + (0.3 * match_ratio)  # Score between 0.7-1.0 based on coverage
        
        return max(regular_sim, no_space_sim, substring_sim)
    else:
        str1 = normalize_name(str1)
        str2 = normalize_name(str2)
        
        if not str1 or not str2:
            return 0
        
        # Use sequence matcher for overall similarity
        return SequenceMatcher(None, str1, str2).ratio()

# Function to find top 2 matches for a name
def find_top_matches(speaker_name, mp_data, mp_eng_name_col_idx, mp_hindi_name_col_idx):
    matches = []
    
    for _, mp_row in mp_data.iterrows():
        eng_name = mp_row.iloc[mp_eng_name_col_idx]  # MP name (English)
        hindi_name = mp_row.iloc[mp_hindi_name_col_idx]  # MP name (Hindi)
        
        # Check word matches for English name
        eng_score, eng_matched_words = check_name_words_match(speaker_name, eng_name, is_hindi=False)
        
        # Check word matches for Hindi name with special handling for Hindi
        hindi_score, hindi_matched_words = check_name_words_match(speaker_name, hindi_name, is_hindi=True) if isinstance(hindi_name, str) else (0, [])
        
        # Add sequence similarity as a tiebreaker
        eng_string_sim = calculate_string_similarity(speaker_name, eng_name, is_hindi=False)
        hindi_string_sim = calculate_string_similarity(speaker_name, hindi_name, is_hindi=True) if isinstance(hindi_name, str) else 0
        
        # Combine scores with weights
        eng_combined = eng_score * 0.8 + eng_string_sim * 0.2
        hindi_combined = hindi_score * 0.8 + hindi_string_sim * 0.2
        
        # Determine which match is better (English or Hindi)
        if eng_combined >= hindi_combined:
            score = eng_combined
            matched_words = eng_matched_words
            used_hindi = False
        else:
            score = hindi_combined
            matched_words = hindi_matched_words
            used_hindi = True
        
        # Add match details
        matches.append({
            'score': score,
            'eng_name': eng_name,
            'hindi_name': hindi_name,
            'matched_words': matched_words,
            'word_count': len(matched_words),
            'string_sim': eng_string_sim if not used_hindi else hindi_string_sim
        })
    
    # Sort matches by:
    # 1. Number of matched words (descending)
    # 2. String similarity (descending) for tiebreakers
    matches.sort(key=lambda x: (-x['word_count'], -x['string_sim'], -x['score']))
    
    # Return top 2 matches
    top_matches = []
    for match in matches[:2]:
        top_matches.append((match['score'], match['eng_name'], match['hindi_name']))
    
    # If there's only one match or second match has zero similarity, duplicate the first
    if len(top_matches) < 2:
        if len(top_matches) > 0:
            top_matches = [top_matches[0], top_matches[0]]
        else:
            top_matches = [(0, "", ""), (0, "", "")]
    
    return top_matches

# Main function
def match_mp_names(mp_file_path, speech_file_path, output_path, 
                  mp_eng_name_col_idx=2, mp_hindi_name_col_idx=3, speech_speaker_col_idx=2):
    """
    Match MP names from the MP list to speakers in speech data.
    
    Parameters:
    - mp_file_path: Path to the MP data file (CSV or XLSX)
    - speech_file_path: Path to the speech data file (CSV)
    - output_path: Path to save the output CSV file
    - mp_eng_name_col_idx: Column index for MP name in English (default: 2 for "mp name")
    - mp_hindi_name_col_idx: Column index for MP name in Hindi (default: 3 for "mp name(hindi)")
    - speech_speaker_col_idx: Column index for speaker name in speech data (default: 2 for "speaker")
    """
    try:
        print(f"Reading MP data from {mp_file_path}...")
        # Read MP data
        if mp_file_path.endswith('.xlsx'):
            mp_data = pd.read_excel(mp_file_path)
        else:
            mp_data = pd.read_csv(mp_file_path)
        
        print(f"Reading speech data from {speech_file_path}...")
        # Read speech data
        speech_data = pd.read_csv(speech_file_path)
        
        print(f"Found {len(mp_data)} MPs and {len(speech_data)} speech entries")
        
        # Create a new dataframe for output
        result_df = speech_data.copy()
        
        # Insert new columns after the speaker column
        new_columns = ['eng name(pref 1)', 'hind name(pref 1)', 'eng name(pref 2)', 'hind name(pref 2)']
        for i, col_name in enumerate(new_columns):
            result_df.insert(speech_speaker_col_idx + 1 + i, col_name, "")
        
        # Process each row in speech data
        total_rows = len(speech_data)
        print(f"Processing {total_rows} speech entries...")
        
        for idx, row in speech_data.iterrows():
            # Get speaker name using column index
            speaker_name = row.iloc[speech_speaker_col_idx]
            
            # Handle NULL/None values properly
            if pd.isna(speaker_name):
                continue
                
            if speaker_name:
                # Check if the speaker is the Speaker/Chair of the House
                if is_speaker_chair(speaker_name):
                    # Set special values for Speaker/Chair
                    if isinstance(speaker_name, str) and ('माननीय अध्यक्ष' in speaker_name or 'अध्यक्ष' in speaker_name):
                        # Hindi version of Speaker
                        result_df.at[idx, 'eng name(pref 1)'] = "HON. SPEAKER"
                        result_df.at[idx, 'hind name(pref 1)'] = "माननीय अध्यक्ष"
                        result_df.at[idx, 'eng name(pref 2)'] = "HON. SPEAKER"
                        result_df.at[idx, 'hind name(pref 2)'] = "माननीय अध्यक्ष"
                    else:
                        # English version of Speaker
                        result_df.at[idx, 'eng name(pref 1)'] = "HON. SPEAKER"
                        result_df.at[idx, 'hind name(pref 1)'] = "माननीय अध्यक्ष"
                        result_df.at[idx, 'eng name(pref 2)'] = "HON. SPEAKER"
                        result_df.at[idx, 'hind name(pref 2)'] = "माननीय अध्यक्ष"
                        
                elif is_chair_chair(speaker_name):
                    if isinstance(speaker_name, str) and ('माननीय सभापति' in speaker_name or 'सभापति' in speaker_name):
                        # Hindi version of Speaker
                        result_df.at[idx, 'eng name(pref 1)'] = "HON. CHAIRPERSON"
                        result_df.at[idx, 'hind name(pref 1)'] = "माननीय सभापति"
                        result_df.at[idx, 'eng name(pref 2)'] = "HON. CHAIRPERSON"
                        result_df.at[idx, 'hind name(pref 2)'] = "माननीय सभापति"
                    else:
                        # English version of Speaker
                        result_df.at[idx, 'eng name(pref 1)'] = "HON. CHAIRPERSON"
                        result_df.at[idx, 'hind name(pref 1)'] = "माननीय सभापति"
                        result_df.at[idx, 'eng name(pref 2)'] = "HON. CHAIRPERSON"
                        result_df.at[idx, 'hind name(pref 2)'] = "माननीय सभापति"
                else:
                    # Regular MP name matching
                    top_matches = find_top_matches(
                        speaker_name, 
                        mp_data, 
                        mp_eng_name_col_idx, 
                        mp_hindi_name_col_idx
                    )
                    
                    # Assign matches to the result dataframe
                    result_df.at[idx, 'eng name(pref 1)'] = top_matches[0][1]
                    result_df.at[idx, 'hind name(pref 1)'] = top_matches[0][2]
                    result_df.at[idx, 'eng name(pref 2)'] = top_matches[1][1]
                    result_df.at[idx, 'hind name(pref 2)'] = top_matches[1][2]
            
            # Print progress
            if (idx + 1) % 100 == 0 or idx == total_rows - 1:
                print(f"Progress: {idx + 1}/{total_rows} entries processed ({(idx + 1)/total_rows*100:.1f}%)")
        
        # Save to CSV
        print(f"Saving results to {output_path}...")
        result_df.to_csv(output_path, index=False)
        
        print(f"Processing complete. Output saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Command line interface
if __name__ == "__main__":
    print(f"Starting MP name matching process...")
    
    # Default file paths and column indices
    mp_file_path = "17th_Lok_Sabha_Members_new.xlsx"
    speech_file_path = "17-III-07.02.2020.csv"
    output_path = "matched_speeches.csv"
    
    # Default column indices (0-based)
    # For MP data: slno(0), constituency(1), mp name(2), mp name(hindi)(3), party(4), state(5)
    mp_eng_name_col_idx = 2
    mp_hindi_name_col_idx = 3
    
    # For speech data: page(0), metadata(1), speaker(2), speech(3)
    speech_speaker_col_idx = 2
    
    # Process command line arguments
    if len(sys.argv) > 1:
        mp_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        speech_file_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    if len(sys.argv) > 4:
        mp_eng_name_col_idx = int(sys.argv[4])
    if len(sys.argv) > 5:
        mp_hindi_name_col_idx = int(sys.argv[5])
    if len(sys.argv) > 6:
        speech_speaker_col_idx = int(sys.argv[6])
    
    print(f"MP data file: {mp_file_path}")
    print(f"Speech data file: {speech_file_path}")
    print(f"Output file: {output_path}")
    print(f"MP English name column index: {mp_eng_name_col_idx}")
    print(f"MP Hindi name column index: {mp_hindi_name_col_idx}")
    print(f"Speech speaker column index: {speech_speaker_col_idx}")
    
    success = match_mp_names(
        mp_file_path, 
        speech_file_path, 
        output_path, 
        mp_eng_name_col_idx, 
        mp_hindi_name_col_idx, 
        speech_speaker_col_idx
    )
    
    if success:
        print("Process completed successfully.")
    else:
        print("Process failed. Please check the error messages above.")