import subprocess
import pandas as pd
from datetime import datetime, timedelta
import os
import tqdm

def download(video_url, duration_in_sec):
    video_id = video_url.split("v=")[1]
    if os.path.exists(f"emirati_videos/{video_id}.webm"):
        return
    command = [
        'yt-dlp',
        video_url,
        '-o', f'emirati_videos/{video_id}.%(ext)s'
    ]

    if duration_in_sec:
        if duration_in_sec > 360:
            command.extend(['--download-sections', f'*0-{duration_in_sec}'])

    
    try:
        subprocess.run(command, check=True)
        print(f"Successfully downloaded {video_url}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
    except FileNotFoundError:
        print("Error: yt-dlp is not installed or not in PATH")

# Example usage
# video_url = "https://www.youtube.com/watch?v=example"
# download(video_url)

if __name__ == "__main__":
    df = pd.read_csv("emirati_playlists_videos_presampled.csv")
    for index, row in tqdm.tqdm(df.iterrows()):
        download(row["video_url"], row["duration_in_sec"])