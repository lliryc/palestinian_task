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



def get_file_parent_folder(drive_service, file_id):
    """Get the parent folder ID of a file."""
    file = drive_service.files().get(fileId=file_id, fields='parents').execute()
    return file.get('parents')[0] if file.get('parents') else None


def set_link_text(service, file_id, link_url):
    """Set the text of a passage in cell C3 of the QnA Survey tab."""
    spreadsheet_id = file_id
    sheet_name = 'Text Correction Task'
    cell_range = 'A3'  # Cell where the hyperlink will be inserted

    # URL and Label for your hyperlink
    url = link_url
    label = 'Click here'

    # Prepare request body (using USER_ENTERED to evaluate formula)
    body = {
        'values': [[f'=HYPERLINK("{url}", "{label}")']]
    }

    # Perform the request
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f'{sheet_name}!{cell_range}',
        valueInputOption='USER_ENTERED',
        body=body
    ).execute()
    

def set_passage_text(service, file_id, passage_text):
    """Set the text of a passage in cells B3 and C3 of the Text Correction Task tab."""
    # Update both B3 and C3 cells with the same passage text
    data = [
        {
            'range': "'Text Correction Task'!B3",
            'values': [[passage_text]]
        },
        {
            'range': "'Text Correction Task'!C3",
            'values': [[passage_text]]
        }
    ]
    
    body = {
        'valueInputOption': 'RAW',
        'data': data
    }
    
    # Use batchUpdate to update both cells in a single API call
    service.spreadsheets().values().batchUpdate(
        spreadsheetId=file_id,
        body=body
    ).execute()

def create_correction_form(file_name:str):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    try:
        # Use service account credentials
        creds = service_account.Credentials.from_service_account_file(
            "google_api_credentials2.json",
            scopes=SCOPES
        )

        # The ID and range of a sample spreadsheet.
        TEMPLATE_SPREADSHEET_ID="1fcBSjUyz7npwh6QTLfTcpad-u7NIyoSKCvRUT_Udf74"

        
        service = build("sheets", "v4", credentials=creds)
        drive_service = build("drive", "v3", credentials=creds)     
        
        # Create a copy of the spreadsheet
        copy_title = file_name # Simplified name
        copy_file = {
            'name': copy_title,
            'parents': [get_file_parent_folder(drive_service, TEMPLATE_SPREADSHEET_ID)]
        }
        
        copied_file = drive_service.files().copy(
            fileId=TEMPLATE_SPREADSHEET_ID,
            body=copy_file
        ).execute()
        
        # Make the file accessible to anyone with the link as an editor
        permission = {
            'type': 'anyone',
            'role': 'writer',
            'allowFileDiscovery': False
        }
        drive_service.permissions().create(
            fileId=copied_file['id'],
            body=permission
        ).execute()
        
        # Generate the shareable link
        share_link = f"https://docs.google.com/spreadsheets/d/{copied_file['id']}/edit"
        
        # Use the new spreadsheet ID for subsequent operations
        file_id = copied_file['id']
        
        return (share_link, file_id)
        
    except Exception as e:
        print(f"Error: {e}")

def get_processed_txt_files():
    with open("processed_correction_forms.txt", "r") as processed_file:
        return set(line.strip() for line in processed_file)

if __name__ == "__main__":
    txt_files = list(glob.glob("gulf_passages/*.txt"))
    filtered_files = filter_files(txt_files)
    ranked_files = rank_files_by_std(filtered_files)
    txt_files = ranked_files[:5]

    processed_txt_files = get_processed_txt_files()
    creds = service_account.Credentials.from_service_account_file(
        "google_api_credentials2.json",
        scopes=SCOPES
    )
    service = build("sheets", "v4", credentials=creds)
    for txt_file in txt_files:
        if txt_file in processed_txt_files:
            print(f"Skipping {txt_file} because it has already been processed")
            continue
        with open(txt_file, "r") as f:
            passage_text = f.read()
        file_name = txt_file.split("/")[-1].split(".")[0]
        
        (link, file_id) = create_correction_form(f"Gulf_Correction_Form_{file_name}")
        set_passage_text(service, file_id, passage_text)
        video_link = f"https://www.youtube.com/watch?v={file_name}"
        set_link_text(service, file_id, video_link)
        with open("correction_registry.txt", "a") as registry_file:
            registry_file.write(f"{link}\t{file_id}\n")
        print(f"Created questionary for {file_name}")
        with open("processed_correction_forms.txt", "a") as processed_file:
            processed_file.write(f"{txt_file}\n") 
        time.sleep(1.2)

