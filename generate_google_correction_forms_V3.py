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


def copy_sheet_within_spreadsheet(service, spreadsheet_id, source_sheet_name, new_sheet_name):
    """
    Copies a sheet within the same spreadsheet using the source sheet name.
    
    Args:
        service: The Google Sheets API service
        spreadsheet_id: ID of the spreadsheet
        source_sheet_name: Name of the sheet to copy
        new_sheet_name: Name for the new sheet
        
    Returns:
        The ID of the newly created sheet
    """
    # First get the sheet ID from the sheet name
    source_sheet_id = get_sheet_id_by_name(service, spreadsheet_id, source_sheet_name)
    
    request = {
        'duplicateSheet': {
            'sourceSheetId': source_sheet_id,
            'insertSheetIndex': 0,  # Insert at the beginning
            'newSheetName': new_sheet_name
        }
    }
    
    body = {
        'requests': [request]
    }
    
    response = service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body
    ).execute()
    
    # Return the new sheet ID
    return response['replies'][0]['duplicateSheet']['properties']['sheetId']

def set_link_text(service, file_id, link_url, sheet_name):
    """Set the text of a passage in cell C3 of the QnA Survey tab."""
    spreadsheet_id = file_id
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

def build_data_for_paragraphs_text(speakers, paragraphs, start_row_number, sheet_name):
    # Use this for values update
    data = []
    current_row_number = start_row_number
    for id, paragraph_list in enumerate(paragraphs):
        speaker = speakers[id]

        if paragraph_list == []:
            continue

        data.append({
                'range': f"'{sheet_name}'!A{current_row_number}",
                'values': [[speaker]]
            })
        
        current_row_number += 1

        for paragraph in paragraph_list:
            
            data.append({
                'range': f"'{sheet_name}'!A{current_row_number}",
                'values': [[paragraph]]
            })
            
            tokens = simple_word_tokenize(paragraph)
            
            for token in tokens:
                data.append({
                    'range': f"'{sheet_name}'!B{current_row_number}",
                    'values': [[token]]
                })
                data.append({
                    'range': f"'{sheet_name}'!C{current_row_number}",
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

def set_passage_text(service, file_id, passage_text, sheet_name):
    """Set the text of a passage in cells B4 and C4 of the Text Correction Task tab."""
    speakers, texts = process_speaker_text(passage_text)
    paragraphs = []
    for text in texts:
        # Remove phrases in square brackets - handles any text except closing bracket
        text = re.sub(r'\[[^\]]*\]', '', text)
        paragraphs.append(split_text_into_paragraphs(text))
    
    data, end_row_number = build_data_for_paragraphs_text(speakers, paragraphs, 4, sheet_name)
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
    sheet_id = get_sheet_id_by_name(service, file_id, sheet_name)
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
        },
        # Add protected range for columns A and B
        {
            "addProtectedRange": {
                "protectedRange": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": end_row_number-1,
                        "startColumnIndex": 0,  # Column A
                        "endColumnIndex": 2     # Up to but not including column C
                    },
                    "description": "Protect original text",
                    "warningOnly": False,
                    "editors": {
                        "users": [
                            # Add service account email if needed
                            "dmitr.belkov@gmail.com",
                            "survey-service@trusty-obelisk-450110-k3.iam.gserviceaccount.com"
                        ],
                        "domainUsersCanEdit": False
                    }
                }
            }
        },
        # Add protected range for cells C1:E3
        {
            "addProtectedRange": {
                "protectedRange": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 3,       # Rows 1-3
                        "startColumnIndex": 2,  # Column C
                        "endColumnIndex": 5     # Up to but not including column F
                    },
                    "description": "Protect header information",
                    "warningOnly": False,
                    "editors": {
                        "users": [
                            "dmitr.belkov@gmail.com",
                            "survey-service@trusty-obelisk-450110-k3.iam.gserviceaccount.com"
                        ],
                        "domainUsersCanEdit": False
                    }
                }
            }
        }
    ]
    service.spreadsheets().batchUpdate(
        spreadsheetId=file_id,
        body={"requests": requests}
    ).execute()

def create_tabs(service, spreadsheet_id, file_names):
    # Reverse the file_names list to process them in reverse order
    reversed_file_names = file_names[::-1]
    sheet_names = [f"ar-ae_{len(reversed_file_names) - i}_{file_name}" for i, file_name in enumerate(reversed_file_names)]
    for sheet_name in sheet_names:
        copy_sheet_within_spreadsheet(service, spreadsheet_id, "Text Correction Task", sheet_name)
    return sheet_names

def create_correction_form(file_name):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """

    # Use service account credentials
    creds = service_account.Credentials.from_service_account_file(
        "google_api_credentials2.json",
        scopes=SCOPES
    )

    # The ID and range of a sample spreadsheet.
    TEMPLATE_SPREADSHEET_ID="1yaUJxZL5z2yBOZqR-TFHEbxiHLRaunwFunpWL4Tdys4"

    
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
        

def get_processed_batches():
    with open("processed_correction_forms_V3.txt", "r") as processed_file:
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

videos_df = pd.read_csv("syrian_playlists_videos_presampled.csv")

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

def delete_sheet_by_name(service, spreadsheet_id, sheet_name):
    """
    Deletes a sheet from a spreadsheet by its name.
    
    Args:
        service: The Google Sheets API service
        spreadsheet_id: ID of the spreadsheet
        sheet_name: Name of the sheet to delete
        
    Returns:
        True if deletion was successful, False if sheet not found
    """

    # First, get the sheet ID by name
    sheet_id = get_sheet_id_by_name(service, spreadsheet_id, sheet_name)
    
    # Prepare the delete request
    request = {
        'deleteSheet': {
            'sheetId': sheet_id
        }
    }
    
    body = {
        'requests': [request]
    }
    
    # Execute the request
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body
    ).execute()
    
    return True
    


def set_active_sheet(service, spreadsheet_id, sheet_name):
    # Get the sheet ID by name
    sheet_id = get_sheet_id_by_name(service, spreadsheet_id, sheet_name)
    
    # Prepare the update properties request
    request = {
        'updateSpreadsheetProperties': {
            'properties': {
                'activeSheetId': sheet_id
            },
            'fields': 'activeSheetId'
        }
    }
    
    body = {
        'requests': [request]
    }
    
    # Execute the request
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body
    ).execute()
    
    return True
    

if __name__ == "__main__":

    txt_files = list(glob.glob("syrian_passages4/*.txt"))
    
    filtered_files = filter_files(txt_files)
    
    ranked_files = rank_files_by_std(filtered_files)
    
    ranked_files = ranked_files[:40]

    # Create list of lists, each containing 100 files
    batch_size = 10

    txt_files_batches = [ranked_files[i:i+batch_size] for i in range(0, len(ranked_files), batch_size)]
    
    get_processed_batches = get_processed_batches()

    creds = service_account.Credentials.from_service_account_file(
        "google_api_credentials2.json",
        scopes=SCOPES
    )
    service = build("sheets", "v4", credentials=creds)

    # Process each batch of files
    for batch_idx0, txt_files in enumerate(txt_files_batches):

        batch_idx = batch_idx0
        print(f"Processing batch {batch_idx+1} of {len(txt_files_batches)}")

        if str(batch_idx) in get_processed_batches:
            print(f"Skipping batch {batch_idx+1} because it has already been processed")
            continue

        file_names = []

        for i, txt_file in enumerate(txt_files):
            file_name = txt_file.split("/")[-1].split(".")[0]
            file_names.append(file_name)

        (link, spreadsheet_id) = create_correction_form(f"{batch_idx+1}_Syrian_Correction_Form_V3")

        sheet_names = create_tabs(service, spreadsheet_id, file_names)

        delete_sheet_by_name(service, spreadsheet_id, "Text Correction Task")

        file_names = []

        for i, txt_file in enumerate(txt_files):

            with open(txt_file, "r") as f:
                passage_text = f.read()

            file_name = txt_file.split("/")[-1].split(".")[0]

            file_names.append(file_name)

            video_link = find_video_url(file_name)

            if video_link is None:
                print(f"No video URL found for {file_name}")
                raise Exception(f"No video URL found for {file_name}")
        
            sheet_name = sheet_names[i]            
            
            set_passage_text(service, spreadsheet_id, passage_text, sheet_name=sheet_name)
            
            set_link_text(service, spreadsheet_id, video_link, sheet_name=sheet_name)
     
        with open("correction_registry_V3.txt", "a") as registry_file:
            registry_file.write(f"{link}\n")

        print(f"Created questionary for {file_names}")
        
        with open("processed_correction_forms_V3.txt", "a") as processed_file:
            processed_file.write(f"{batch_idx}\n") 
            time.sleep(1.2)
