import os.path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import glob
import time
import os
import glob
import re
import numpy as np
import dotenv
import fasttext
from huggingface_hub import hf_hub_download

dotenv.load_dotenv()

# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive"
]

model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model = fasttext.load_model(model_path)

pattern = r"\*\*\s\d\s[\w]+\s\*\*"

def count_speakers(text):
    options =  re.findall(pattern, text)
    return len(set(options))


def get_files_4more_speakers(files):

    files_4more_speakers = []    
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            if count_speakers(text) >= 4:
                files_4more_speakers.append(file)
    return files_4more_speakers

speaker_line = r"\[\d+ - \d+\]  \*\* \d+ المتحدث \*\*"

def detect_non_arabic(text):
    text = re.sub(speaker_line, '', text)
    text = text.replace('\n', ' ')
    text = text.strip()
    total_cnt = 0
    non_arabic_cnt = 0
    tokens = text.split(' ')
    for i in range(0, len(tokens), 20):
        chunk = tokens[i:i+20]
        chunk_str = ' '.join(chunk)
        chunk_str = chunk_str.strip()
        if chunk_str == '':
            continue
        pred = model.predict([chunk_str], k=1)
        res = str(pred[0][0][0])
        if '_Arab' not in res:
            non_arabic_cnt += 1
        total_cnt += 1
    if non_arabic_cnt / total_cnt > 0.3:
        return True
    return False

def get_files_non_arabic_speech(files):
    files_non_arabic_speech = []    
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            if detect_non_arabic(text):
                files_non_arabic_speech.append(file)
    return files_non_arabic_speech

def detect_msa(text):
    text = re.sub(speaker_line, '', text)
    text = text.replace('\n', ' ')
    text = text.strip()
    total_cnt = 0
    msa_cnt = 0
    tokens = text.split(' ')
    for i in range(0, len(tokens), 20):
        chunk = tokens[i:i + 20]
        chunk_str = ' '.join(chunk)
        chunk_str = chunk_str.strip()
        if chunk_str == '':
            continue
        pred = model.predict([chunk_str], k=1)
        res = str(pred[0][0][0])
        if '_arb_' in res:
            msa_cnt += 1
        total_cnt += 1
    if msa_cnt / total_cnt > 0.6:
        return True
    return False

def get_files_msa_speech(files):
    files_msa_speech = []    
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            if detect_msa(text):
                files_msa_speech.append(file)
    return files_msa_speech

def filter_files(files):
    files_4more_speakers = get_files_4more_speakers(files)
    files_non_arabic_speech = get_files_non_arabic_speech(files)
    files_msa_speech = get_files_msa_speech(files)
    not_include = set(files_4more_speakers + files_non_arabic_speech + files_msa_speech)
    filtered_files = [file for file in files if file not in not_include]
    return filtered_files


def split_by_speaker(text):
    blocks = re.split(f"({speaker_line})", text.strip())
    return [block.strip() for block in blocks[2::2] if block.strip()]

def avg_std_ch(text):
    segments = split_by_speaker(text)
    data_series = []
    if len(segments) == 1:
        segments.append('')
    for segment in segments:
        data_series.append(len(segment))
    data_series = np.array(data_series)
    mean = np.mean(data_series)
    std = np.std(data_series)
    return mean, std

def rank_files_by_std(files):
    files_stds = []        
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            _, std = avg_std_ch(text)            
            files_stds.append(std)

    files_stds = list(zip(files, files_stds))
    files_sorted_by_std_desc = [file for file, _ in sorted(files_stds, key=lambda x: x[1], reverse=True)]
    return files_sorted_by_std_desc



def set_passage_text(service, file_id, passage_text, row_number):
    """Set the text of a passage in cell A{n} of the Sheet1 tab."""
    # Prepare the value update request
    range_name = f"'Sheet1'!A{row_number+2}"
    value_range_body = {
        'values': [[passage_text]]  # Double array as required by Sheets API
    }
    
    # Update the cell value
    service.spreadsheets().values().update(
        spreadsheetId=file_id,
        range=range_name,
        valueInputOption='RAW',
        body=value_range_body
    ).execute()

# The ID and range of a sample spreadsheet.
#GOOGLE_SPREADSHEET_ID="1QmbJ9PDLXa42sJVTvsnWVuq_qgPaoZkcrHh38F1j8O8"

GOOGLE_SPREADSHEET_ID="1DDXQmGO6aZ1vyet3BgNP-l3LuJdDiKDHByx4ggWjI3w"

if __name__ == "__main__":

    txt_files = list(glob.glob("palestinian_passages/*.txt"))
    filtered_files = filter_files(txt_files)
    ranked_files = rank_files_by_std(filtered_files)
    txt_files = ranked_files

    creds = service_account.Credentials.from_service_account_file(
        "google_api_credentials2.json",
        scopes=SCOPES
    )
    service = build("sheets", "v4", credentials=creds)
    passage_texts = []
    file_names = []

    for i, txt_file in enumerate(txt_files):
        with open(txt_file, "r") as f:
            passage_text = f.read()
        passage_texts.append(passage_text)
        file_name = txt_file.split("/")[-1].split(".")[0]        
        file_names.append(file_name)

    files_passages = list(zip(file_names, passage_texts))

    for i, (file_name, passage_text) in enumerate(files_passages):
        link = f"https://www.youtube.com/watch?v={file_name}"
        passage_text_link = f"{link}\n\n{passage_text}"
        set_passage_text(service, GOOGLE_SPREADSHEET_ID, passage_text_link, i)
        print(f"Set passage text for {file_name}")
        time.sleep(2)

