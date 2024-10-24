import csv
import re

def extract_speech(file_path):
    # Open the input text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into lines
    lines = text.splitlines()

    # Regular expression to identify valid speaker names (up to first colon on each line)
    speaker_pattern = re.compile(r'^([^:]+):\s*(.*)$')

    # To store the extracted speaker-speech pairs
    data = []
    current_speaker = None
    current_speech = []

    # Flag to indicate whether we've encountered the first valid speaker
    found_first_speaker = False

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Check if the line starts with a speaker's name followed by a colon
        match = speaker_pattern.match(line)

        if match:
            speaker = match.group(1).strip()
            speech = match.group(2).strip()

            # Check if the speaker has fewer than 5 words in the name
            if len(speaker.split()) < 5:
                # If we haven't found the first speaker yet, ignore all prior lines
                found_first_speaker = True

                # If there's already a current speaker, append their speech to the data
                if current_speaker:
                    data.append((current_speaker, " ".join(current_speech)))

                # Start new speaker and speech
                current_speaker = speaker
                current_speech = [speech]
            else:
                # If it's not a valid speaker, consider it as part of the speech
                if found_first_speaker and current_speaker:
                    current_speech.append(line.strip())
        else:
            # If no speaker is found and we're past the first valid speaker, treat the line as speech
            if found_first_speaker and current_speaker:
                current_speech.append(line.strip())

    # Add the last speaker-speech pair to the data
    if current_speaker:
        data.append((current_speaker, " ".join(current_speech)))

    # Write the data to a CSV file
    with open('table.csv', 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Name', 'Speech'])

        for name, speech in data:
            writer.writerow([name, speech])

    print("Data has been successfully written to table.csv")

# Specify the input text file
input_file = 'extracted_text3.txt'
extract_speech(input_file)
