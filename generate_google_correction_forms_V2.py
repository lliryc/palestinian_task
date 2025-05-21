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
from tokenize2 import simple_word_tokenize
import pandas as pd
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
    cell_range = 'A2'  # Cell where the hyperlink will be inserted

    # URL and Label for your hyperlink
    url = link_url
    label = 'Click here to watch the video'

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
    

def build_data_for_paragraphs_text(speakers, paragraphs, start_row_number):
    # Use this for values update
    data = []
    current_row_number = start_row_number
    for id, paragraph_list in enumerate(paragraphs):
        speaker = speakers[id]

        if paragraph_list == []:
            continue

        data.append({
                'range': f"'Text Correction Task'!A{current_row_number}",
                'values': [[speaker]]
            })
        
        current_row_number += 1

        for paragraph in paragraph_list:
            
            data.append({
                'range': f"'Text Correction Task'!A{current_row_number}",
                'values': [[paragraph]]
            })
            
            tokens = simple_word_tokenize(paragraph)
            
            for token in tokens:
                data.append({
                    'range': f"'Text Correction Task'!B{current_row_number}",
                    'values': [[token]]
                })
                data.append({
                    'range': f"'Text Correction Task'!C{current_row_number}",
                    'values': [[token]]
                })
                current_row_number += 1
            current_row_number += 1
    return data, current_row_number

def get_sheet_id_by_name(service, spreadsheet_id, sheet_name):
    """Get the sheetId for a given sheet name."""
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sheet in spreadsheet['sheets']:
        if sheet['properties']['title'] == sheet_name:
            return sheet['properties']['sheetId']
    raise ValueError(f"Sheet name '{sheet_name}' not found.")

def set_passage_text(service, file_id, passage_text):
    """Set the text of a passage in cells B4 and C4 of the Text Correction Task tab."""
    speakers, texts = process_speaker_text(passage_text)
    paragraphs = []
    for text in texts:
        # Remove phrases in square brackets - handles any text except closing bracket
        text = re.sub(r'\[[^\]]*\]', '', text)
        paragraphs.append(split_text_into_paragraphs(text))
    
    data, end_row_number = build_data_for_paragraphs_text(speakers, paragraphs, 4)
    # First update values
    value_body = {
        'valueInputOption': 'RAW',
        'data': data
    } 
    
    service.spreadsheets().values().batchUpdate(
        spreadsheetId=file_id,
        body=value_body
    ).execute()

    # Set text direction to left-to-right for columns A, B, and C, starting from row 4
    sheet_id = get_sheet_id_by_name(service, file_id, 'Text Correction Task')
    requests = [
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 3,  # Row 4 (0-based index)
                    "endRowIndex": end_row_number-1,
                    "startColumnIndex": 0,
                    "endColumnIndex": 3
                },
                "cell": {
                    "userEnteredFormat": {
                        "textDirection": "RIGHT_TO_LEFT"
                    }
                },
                "fields": "userEnteredFormat.textDirection"
            }
        },
        # Add conditional formatting to highlight cells where B and C don't match
        {
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [
                        {
                            "sheetId": sheet_id,
                            "startRowIndex": 3,  # Row 4 (0-based index)
                            "endRowIndex": end_row_number-1,
                            "startColumnIndex": 1,  # Column B
                            "endColumnIndex": 2    # Up to column C
                        }
                    ],
                    "booleanRule": {
                        "condition": {
                            "type": "CUSTOM_FORMULA",
                            "values": [
                                {
                                    "userEnteredValue": "=NOT(EXACT(B4:B,C4:C))"
                                }
                            ]
                        },
                        "format": {
                            "backgroundColor": {
                                "red": 1.0,
                                "green": 0.9,
                                "blue": 0.9
                            }
                        }
                    }
                },
                "index": 0
            }
        },
                {
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [
                        {
                            "sheetId": sheet_id,
                            "startRowIndex": 3,  # Row 4 (0-based index)
                            "endRowIndex": end_row_number-1,
                            "startColumnIndex": 2,  # Column B
                            "endColumnIndex": 3    # Up to column C
                        }
                    ],
                    "booleanRule": {
                        "condition": {
                            "type": "CUSTOM_FORMULA",
                            "values": [
                                {
                                    "userEnteredValue": "=NOT(EXACT(B4:B,C4:C))"
                                }
                            ]
                        },
                        "format": {
                            "backgroundColor": {
                                "red": 0.9,
                                "green": 1.0,
                                "blue": 0.9
                            }
                        }
                    }
                },
                "index": 0
            }
        }
    ]
    service.spreadsheets().batchUpdate(
        spreadsheetId=file_id,
        body={"requests": requests}
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
        TEMPLATE_SPREADSHEET_ID="1I1ib4Xjp_rz6A-s-_Lyywl1IBnEr85_jQ9KwIQ3xYlA"

        
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
    with open("processed_correction_forms_V2.txt", "r") as processed_file:
        return set(line.strip() for line in processed_file)

def process_speaker_text(file_content):
    """
    Splits text content into speaker labels and their corresponding text
    
    Args:
        file_content (str): The content of the text file
        
    Returns:
        tuple: A tuple containing (speakers, texts) lists
    """
    # Regular expression to match speaker labels and text
    pattern = r'(\[\d+ - \d+\]\s+\*\* \d+ المتحدث \*\*)\s*([\s\S]*?)(?=\[\d+ - \d+\]|\Z)'
    
    speakers = []
    texts = []
    
    # Find all matches in the file content
    matches = re.finditer(pattern, file_content)
    
    for match in matches:
        # Extract speaker number
        speaker_number = match.group(1)
        # Extract and trim text content
        text_content = match.group(2).strip()
        
        speakers.append(speaker_number)
        texts.append(text_content)
    
    return speakers, texts

def split_text_into_paragraphs(text):
    paragraphs = re.split(r'\n', text)
    return [p for p in paragraphs if p.strip()]

videos_df = pd.read_csv("gulf_channels_videos_presampled.csv")

def find_video_url(file_name):
    # Check if the DataFrame has the required column
    if 'video_url' not in videos_df.columns:
        print(f"Warning: 'video_url' column not found in DataFrame for {file_name}")
        return None
        
    
    # Find rows where video_url contains the file_name
    matching_rows = videos_df[videos_df['video_url'].str.contains(file_name, na=False)]
    
    if not matching_rows.empty:
        # Return the first matching URL
        return matching_rows.iloc[0]['video_url']
    else:
        print(f"No matching video URL found for {file_name}")
        return None

if __name__ == "__main__":
    txt_files = list(glob.glob("gulf_passages/*.txt"))
    filtered_files = filter_files(txt_files)
    ranked_files = rank_files_by_std(filtered_files)
    txt_files = ranked_files[:100]

    processed_txt_files = get_processed_txt_files()
    creds = service_account.Credentials.from_service_account_file(
        "google_api_credentials2.json",
        scopes=SCOPES
    )
    service = build("sheets", "v4", credentials=creds)
    for i, txt_file in enumerate(txt_files):
        if txt_file in processed_txt_files:
            print(f"Skipping {txt_file} because it has already been processed")
            continue
        with open(txt_file, "r") as f:
            passage_text = f.read()
        file_name = txt_file.split("/")[-1].split(".")[0]

        video_link = find_video_url(file_name)

        if video_link is None:
            print(f"No video URL found for {file_name}")
            continue
        
        (link, file_id) = create_correction_form(f"{i+1}_Gulf_Correction_Form_V2_{file_name}")
        
        set_passage_text(service, file_id, passage_text)
        
        set_link_text(service, file_id, video_link)
        
        with open("correction_registry_V2.txt", "a") as registry_file:
            registry_file.write(f"{link}\n")
        print(f"Created questionary for {file_name}")
        with open("processed_correction_forms_V2.txt", "a") as processed_file:
            processed_file.write(f"{txt_file}\n") 
        time.sleep(1.2)
