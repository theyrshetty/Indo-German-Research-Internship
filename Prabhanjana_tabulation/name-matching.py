import pandas as pd
import re
from collections import Counter
import langdetect

def process_csv(input_file, output_file):
    """
    Process the CSV file to determine the correct speaker and name information
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the processed CSV file
    """
    print(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Ensure the required columns exist
    required_columns = ['page', 'metadata', 'speaker', 
                       'eng name(pref 1)', 'hind name(pref 1)', 
                       'eng name(pref 2)', 'hind name(pref 2)', 
                       'speech']
    
    # Standardize column names by removing whitespace
    df.columns = [col.strip() for col in df.columns]
    
    # Map the actual column names to expected ones if needed
    column_mapping = {}
    for req_col in required_columns:
        for actual_col in df.columns:
            if req_col.lower().replace(' ', '') == actual_col.lower().replace(' ', ''):
                column_mapping[req_col] = actual_col
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Step 1: Count frequency of speakers in both preference sets
    pref1_speakers_count = Counter(df['eng name(pref 1)'].fillna('Unknown'))
    pref2_speakers_count = Counter(df['eng name(pref 2)'].fillna('Unknown'))
    
    # Get the top 4 most occurring names in preference 1
    top_speakers = [name for name, _ in pref1_speakers_count.most_common(4) if name != 'Unknown']
    
    # Identify potential alternating pattern (common in dialogues)
    def get_alternating_pattern(sequence):
        # Skip None/NaN values
        filtered_seq = [s for s in sequence if pd.notna(s) and s != '']
        if len(filtered_seq) < 4:  # Need at least 4 entries to detect a pattern
            return None
        
        # Check for A-B-A-B pattern
        patterns = []
        for i in range(0, len(filtered_seq) - 3):
            if (filtered_seq[i] == filtered_seq[i+2] and 
                filtered_seq[i+1] == filtered_seq[i+3] and
                filtered_seq[i] != filtered_seq[i+1]):
                patterns.append((filtered_seq[i], filtered_seq[i+1]))
        
        # Return the most common pattern if found
        if patterns:
            pattern_counts = Counter(patterns)
            return pattern_counts.most_common(1)[0][0]
        
        return None
    
    # Detect patterns in speaker sequence
    pref1_speaker_sequence = df['eng name(pref 1)'].tolist()
    pref2_speaker_sequence = df['eng name(pref 2)'].tolist()
    
    # Create a new DataFrame for the output
    output_df = pd.DataFrame()
    
    # Copy the page, metadata, speaker columns
    output_df['page'] = df['page']
    output_df['metadata'] = df['metadata']
    output_df['speaker'] = df['speaker']
    
    # Initialize output columns
    output_df['eng name'] = ''
    output_df['hindi name'] = ''
    
    # Helper function to detect if text is Hindi
    def is_hindi(text):
        if not isinstance(text, str) or text.strip() == '':
            return False
        
        try:
            # Try to detect the language
            lang = langdetect.detect(text)
            return lang == 'hi'
        except:
            # If detection fails, check for Hindi Unicode range
            hindi_pattern = re.compile(r'[\u0900-\u097F]')
            return bool(hindi_pattern.search(text))
    
    # Identify if original speaker column contains Hindi text for each row
    df['is_hindi_speaker'] = df['speaker'].apply(is_hindi)
    
    # Get the alternating patterns
    pref1_pattern = get_alternating_pattern(pref1_speaker_sequence)
    pref2_pattern = get_alternating_pattern(pref2_speaker_sequence)
    
    # Process each row to determine correct speaker
    # Keep track of the current pattern position
    pattern_position = 0
    
    for i, row in df.iterrows():
        speaker_pref1 = row['eng name(pref 1)']
        speaker_pref2 = row['eng name(pref 2)']
        
        # Decision logic for determining correct speaker
        selected_speaker = None
        
        # If both preferences agree, use that speaker
        if pd.notna(speaker_pref1) and pd.notna(speaker_pref2) and speaker_pref1 == speaker_pref2:
            selected_speaker = speaker_pref1
        
        # Logic Path 1: If a name occurs only once in pref 1, but pref 2 has one of the top 4 most frequent names
        elif (pd.notna(speaker_pref1) and pd.notna(speaker_pref2) and 
            pref1_speakers_count[speaker_pref1] == 1 and speaker_pref2 in top_speakers):
            selected_speaker = speaker_pref2
        
        # Logic Path 2: If original speaker is in Hindi and pref 2 is a prominent speaker
        elif row['is_hindi_speaker'] and pd.notna(speaker_pref2) and speaker_pref2 in top_speakers:
            # Check if this pref2 speaker commonly appears when original text is in English
            eng_speaker_rows = df[~df['is_hindi_speaker']]
            if speaker_pref2 in eng_speaker_rows['eng name(pref 2)'].values:
                selected_speaker = speaker_pref2
            else:
                selected_speaker = speaker_pref1
        
        # Logic Path 3: ABAB pattern recognition
        elif pref1_pattern and pd.notna(speaker_pref1) and pd.notna(speaker_pref2):
            # Calculate expected speaker based on pattern
            expected_speaker = None
            if i > 0 and pd.notna(output_df.at[i-1, 'eng name']):
                last_speaker = output_df.at[i-1, 'eng name']
                if last_speaker == pref1_pattern[0]:
                    expected_speaker = pref1_pattern[1]
                elif last_speaker == pref1_pattern[1]:
                    expected_speaker = pref1_pattern[0]
            
            # If we have an expected speaker and pref1 breaks pattern but pref2 maintains it
            if expected_speaker and speaker_pref1 != expected_speaker and speaker_pref2 == expected_speaker:
                selected_speaker = speaker_pref2
            else:
                selected_speaker = speaker_pref1
                
            # Update pattern position
            if selected_speaker == pref1_pattern[0]:
                pattern_position = 1  # Next expected is pattern[1]
            elif selected_speaker == pref1_pattern[1]:
                pattern_position = 0  # Next expected is pattern[0]
        
        # Default to pref 1 (highest probability of being correct)
        else:
            selected_speaker = speaker_pref1 if pd.notna(speaker_pref1) else speaker_pref2
        
        # If still no valid speaker, use whatever is available
        if not selected_speaker or pd.isna(selected_speaker):
            selected_speaker = speaker_pref1 if pd.notna(speaker_pref1) else speaker_pref2
            if not selected_speaker or pd.isna(selected_speaker):
                selected_speaker = row['speaker'] if pd.notna(row['speaker']) else "Unknown"
        
        # Set the eng name
        output_df.at[i, 'eng name'] = selected_speaker
        
        # Set the corresponding hindi name based on which preference was chosen
        if selected_speaker == speaker_pref1:
            output_df.at[i, 'hindi name'] = row['hind name(pref 1)']
        elif selected_speaker == speaker_pref2:
            output_df.at[i, 'hindi name'] = row['hind name(pref 2)']
        else:
            # If the logic chose a different speaker (rare case), use pref1's Hindi name
            output_df.at[i, 'hindi name'] = row['hind name(pref 1)'] if pd.notna(row['hind name(pref 1)']) else row['hind name(pref 2)']
    
    # Add speech column at the end
    output_df['speech'] = df['speech']
    
    # Save the processed data
    print(f"Saving processed data to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print("Processing complete!")
    
    # Return some statistics
    return {
        "total_rows": len(df),
        "pref1_used": sum(output_df['eng name'] == df['eng name(pref 1)']),
        "pref2_used": sum(output_df['eng name'] == df['eng name(pref 2)']),
        "unique_speakers": len(set(output_df['eng name'].dropna()))
    }

# Main function with hardcoded file paths
def main():
    # Change these file paths to your input and output files
    input_file = "matched_speeches.csv"   # Replace with your input file
    output_file = "outputfile.csv"        # Replace with your output file
    
    try:
        # Run the processing with the specified file names
        stats = process_csv(input_file, output_file)
        
        # Display statistics
        print("\nStatistics:")
        print(f"Total rows processed: {stats['total_rows']}")
        print(f"Rows using preference 1: {stats['pref1_used']} ({stats['pref1_used']/stats['total_rows']*100:.1f}%)")
        print(f"Rows using preference 2: {stats['pref2_used']} ({stats['pref2_used']/stats['total_rows']*100:.1f}%)")
        print(f"Unique speakers identified: {stats['unique_speakers']}")
        
    except Exception as e:
        print(f"Error: {e}")

# Run the main function
if __name__ == "__main__":
    main()