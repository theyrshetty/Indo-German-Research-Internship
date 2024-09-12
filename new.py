import os
import requests
import pandas as pd
from datetime import datetime

class LokSabhaPDFDownloader:
    def __init__(self, csv_file_path, save_directory, start_lok_sabha, end_lok_sabha):
        self.csv_file_path = csv_file_path
        self.save_directory = save_directory
        self.start_lok_sabha = start_lok_sabha
        self.end_lok_sabha = end_lok_sabha
        self.base_url = "https://sansad.in/getFile/debatestextmk/{lok_sabha}/{session}/{date}.pdf?source=loksabhadocs"
        self.error_log_path = os.path.join(self.save_directory, "errors.csv")
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Load the CSV and sort by date
        self.df = self.load_and_sort_csv()
        
        # Read existing error log or create an empty DataFrame
        self.error_log_df = self.load_error_log()

    def load_and_sort_csv(self):
        df = pd.read_csv(self.csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        return df.sort_values(by='Date', ascending=False)

    def load_error_log(self):
        if os.path.exists(self.error_log_path):
            return pd.read_csv(self.error_log_path)
        return pd.DataFrame(columns=['File Name', 'Error'])

    def int_to_roman(self, n):
        roman_numerals = [
            ('M', 1000), ('CM', 900), ('D', 500), ('CD', 400),
            ('C', 100), ('XC', 90), ('L', 50), ('XL', 40),
            ('X', 10), ('IX', 9), ('V', 5), ('IV', 4), ('I', 1)
        ]
        result = ''
        for roman, value in roman_numerals:
            while n >= value:
                result += roman
                n -= value
        return result

    def download_pdf(self, url, save_path):
        try:
            response = requests.get(url, stream=True, verify=False)
            response.raise_for_status()
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            if os.path.getsize(save_path) < 2048:
                os.remove(save_path)
                print(f"File is smaller than 2 KB, deleted: {save_path}")
                return False
            print(f"Downloaded: {save_path}")
            return True
        except (requests.exceptions.HTTPError, requests.exceptions.SSLError) as e:
            print(f"Failed to download {url}: {e}")
            return False

    def construct_urls(self, lok_sabha, session_roman, date):
        urls = [
            self.base_url.format(lok_sabha=lok_sabha, session=session_roman, date=date),
            self.base_url.format(lok_sabha=lok_sabha, session=session_roman, date=f"{date}F"),
            self.base_url.format(lok_sabha=lok_sabha, session=session_roman, date=date.replace('.', '')[:4])  # DDMM format
        ]
        return urls

    def process_row(self, row):
        lok_sabha = row['Lok Sabha']
        session_number = row['Session']
        date = row['Date'].strftime('%d.%m.%Y')
        session_roman = self.int_to_roman(session_number)

        if lok_sabha < self.end_lok_sabha or lok_sabha > self.start_lok_sabha:
            return

        file_name_base = f"{lok_sabha}-{session_roman}-{date}.pdf"
        save_path = os.path.join(self.save_directory, file_name_base)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            print(f"File already exists, skipping: {save_path}")
            return

        if not self.error_log_df[(self.error_log_df['File Name'] == file_name_base)].empty:
            print(f"Already logged error for: {file_name_base}, skipping")
            return

        urls = self.construct_urls(lok_sabha, session_roman, date)

        for url in urls:
            if self.download_pdf(url, save_path):
                return  # Stop after first successful download

        # If all attempts fail, log the error
        self.error_log_df = pd.concat([self.error_log_df, pd.DataFrame({'File Name': [file_name_base], 'Error': ['Failed to download']})], ignore_index=True)

    def run(self):
        for _, row in self.df.iterrows():
            self.process_row(row)
        self.error_log_df.to_csv(self.error_log_path, index=False)
        print("Download process completed.")


csv_file_path = "loksabha_sessions.csv"
save_directory = r"C:\Users\Daksh Vats\OneDrive\Desktop\Research papers 15mincity\webscraper\Downloads"
start_lok_sabha = 18
end_lok_sabha = 16

downloader = LokSabhaPDFDownloader(csv_file_path, save_directory, start_lok_sabha, end_lok_sabha)
downloader.run()
