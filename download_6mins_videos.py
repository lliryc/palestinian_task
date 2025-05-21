import subprocess
import pandas as pd
from datetime import datetime, timedelta
import os
import tqdm
import glob

def download(video_url, duration_in_sec):
    # Extract video_id from different URL formats
    if "youtube.com/watch?v=" in video_url:
        video_id = video_url.split("v=")[1]
        # Handle additional parameters after video ID
        if "&" in video_id:
            video_id = video_id.split("&")[0]
    elif "youtube.com/shorts/" in video_url:
        video_id = video_url.split("shorts/")[1]
        # Handle additional parameters after video ID
        if "?" in video_id:
            video_id = video_id.split("?")[0]
    else:
        print(f"Unsupported URL format: {video_url}")
        return
    
    # Check if any file with this video_id exists (regardless of extension)
    if glob.glob(f"palestinian_videos/{video_id}.*"):
        print(f"Video {video_id} already exists")
        return
    
    command = [
        'yt-dlp',
        video_url,
        '-o', f'palestinian_videos/{video_id}.%(ext)s'
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
    df = pd.read_csv("palestinian_playlists_videos_presampled.csv")
    for index, row in tqdm.tqdm(df.iterrows()):
        download(row["video_url"], row["duration_in_sec"])
