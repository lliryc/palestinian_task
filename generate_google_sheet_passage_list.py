import os.path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import glob
import time

# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive"
]


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
GOOGLE_SPREADSHEET_ID="1b9wWtD-pZInjfPneJEKuSMbEGXJt1xttH_0-2PhVlEQ"

if __name__ == "__main__":
    txt_files = list(glob.glob("gulf_passages/*.txt"))
    creds = service_account.Credentials.from_service_account_file(
        "google_api_credentials2.json",
        scopes=SCOPES
    )
    service = build("sheets", "v4", credentials=creds)
    for i, txt_file in enumerate(txt_files):
        with open(txt_file, "r") as f:
            passage_text = f.read()
        file_name = txt_file.split("/")[-1].split(".")[0]        
        link = f"https://www.youtube.com/watch?v={file_name}"
        passage_text = f"{link}\n\n{passage_text}"
        set_passage_text(service, GOOGLE_SPREADSHEET_ID, passage_text, i)
        print(f"Set passage text for {file_name}")
        time.sleep(2)

