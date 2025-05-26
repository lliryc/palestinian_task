import requests
import time
import pandas as pd
import json
import glob

#curl -X 'GET' \
#  'http://34.154.229.72:9001/transcripts/ar/_2SJIyXlhhc' \
#  -H 'accept: application/json'


def get_transcript(video_id):
    tries = 10
    while tries > 0:
        url = f"http://34.154.73.208:8999/transcripts/ar/{video_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        tries -= 1
        time.sleep(1)
    return None

#def get_processed_transcripts2():
#    with open("processed_transcripts.txt", 'r') as f:
#    return set([line.strip() for line in f.readlines()])

def get_processed_transcripts():
    files = list(glob.glob('palestinian_videos_transcripts/*.json'))
    return list([file.split('/')[-1].split('.')[0] for file in files])

def write_processed_transcript(video_id):
    with open("processed_transcripts.txt", 'a') as f:
        f.write(f"{video_id}\n")
        f.flush()

if __name__ == "__main__":
    
    df = pd.read_csv("palestinian_playlists_videos_presampled.csv")
    
    processed_transcripts = get_processed_transcripts()
    
    for index, row in df.iterrows():
        
        print(f"Processing {index} of {len(df)}")

        time_start = time.time()

        video_url = row["video_url"]
        # Extract video ID from both standard YouTube URLs and shorts URLs
        if "/shorts/" in video_url:
            video_id = video_url.split("/shorts/")[-1]
        else:
            video_id = video_url.split("v=")[-1]
        
        # Remove any additional parameters that might be in the URL
        if "&" in video_id:
            video_id = video_id.split("&")[0]
        
        if video_id in processed_transcripts:
            print(f"Skipping {video_id} because it has already been processed")
            continue
        
        transcript = get_transcript(video_id)
        
        write_processed_transcript(video_id)

        if transcript is None or "transcript" not in transcript:
            print(f"Failed to get transcript for {video_id}")
            continue    

        with open(f"palestinian_videos_transcripts/{video_id}.json", 'w') as f:

            json.dump(transcript["transcript"], f)

        time_end = time.time()
        print(f"Time taken per iteration: {time_end - time_start} seconds")
