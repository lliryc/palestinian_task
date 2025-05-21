import subprocess
import multiprocessing
from pytimeparse.timeparse import timeparse
import pandas as pd
import uuid
import os
from datetime import datetime, timedelta

def fetch_playlist_video_urls(playlist_url):
    """
    Step 1: Fetch the *list of video URLs* for a given playlist.
    Uses --flat-playlist so we only get references (fast).
    """
    command = [
        "yt-dlp",
        "--flat-playlist",
        "--skip-download",
        "--print", "%(webpage_url)s",
        playlist_url
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    # Each line in stdout is a video URL
    urls = result.stdout.strip().split("\n")
    # Filter out any empty lines
    return [u for u in urls if u.strip()]

def fetch_video_metadata(video_url):
    """
    Step 2: For a given *video URL*, extract desired metadata
    (e.g., title, duration, webpage_url) without downloading.
    """
    command = [
        "yt-dlp",
        "--skip-download",
        "--print", "%(duration_string)s\t%(upload_date)s",
        video_url
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()

def presample_videos(sample_size):
    cdf = pd.DataFrame()
    for file in os.listdir("palestinian_playlists_videos"):
        df = pd.read_csv(f"palestinian_playlists_videos/{file}")
        # Convert upload_date to datetime
        df['upload_date'] = pd.to_datetime(df['upload_date'], format='%Y%m%d')

        # Step 1: Filter out rows where duration_in_sec is missing
        df = df.dropna(subset=['duration_in_sec'])

        # Step 2: Filter out videos with duration less than 360 seconds
        #df = df[df['duration_in_sec'] >= 360]

        # Step 3: Filter out videos with upload_date earlier than 5 months ago
        five_months_ago = datetime.now() - timedelta(days=5 * 30)  # Approximate 5 months as 150 days
        df = df[df['upload_date'] >= five_months_ago]
        
        if len(df) > sample_size:
            df = df.sample(n=sample_size)
        else:
            df = df.sample(n=len(df))
        cdf = pd.concat([cdf, df])
        
    print(len(cdf))
    cdf.to_csv("palestinian_playlists_videos_presampled.csv", index=False)


def get_transcripts_metadata():
  
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(script_directory, "palestinian_playlists.txt"), "r") as playlist_file:
        playlist_urls = playlist_file.readlines()
        
    playlist_urls = [url.strip() for url in playlist_urls]
    
    playlist_urls = [url for url in set(playlist_urls)]
  
    with open(os.path.join(script_directory, "palestinian_playlists_processed.txt"), "r") as processed_file:
        processed = processed_file.readlines()
        
    processed = set([url.strip() for url in processed])
    
    playlist_urls = [url for url in playlist_urls if url not in processed]
        
    with open(os.path.join(script_directory, "palestinian_playlists_processed.txt"), "a") as processed_file:
      
        playlists_len = len(playlist_urls)
        
        if playlists_len  > 500:
          playlist_urls = playlist_urls[:500]
          playlists_len = 500
        
        for i, playlist_url in enumerate(playlist_urls):
          
            channel_id = str(uuid.uuid4())
            print(f"Processing playlist: {playlist_url} (#{i} from {playlists_len})")
            
            try:
        
              # --- Step 1: Fetch all video URLs from the playlist
              video_urls = fetch_playlist_video_urls(playlist_url)
              print(f"Found {len(video_urls)} video URLs in playlist.\n")

              # --- Step 2: Use multiprocessing to fetch metadata for each video
              # Create a Pool with 16 workers
              with multiprocessing.Pool(processes=16) as pool:
                  results = pool.map(fetch_video_metadata, video_urls)
              
              results = [result.split("\t") for result in results]
              # Add video URLs to the results
              results = [(video_url, *result) for video_url, result in zip(video_urls, results)]
              emirati_playlist_df = pd.DataFrame(results, columns=["video_url", "duration", "upload_date"])
              emirati_playlist_df["playlist_url"] = playlist_url
              emirati_playlist_df["duration_in_sec"] = emirati_playlist_df["duration"].apply(lambda x: timeparse(x))
              
              emirati_playlist_df.drop(columns=["duration"], inplace=True)
              
              emirati_playlist_df.to_csv(os.path.join(script_directory, "palestinian_playlists_videos", f"{channel_id}.csv"), index=False)
              # Print each metadata line (duration, upload_date)
              for line in results:
                  print(line)
              processed_file.write(f"{playlist_url}\n")
              processed_file.flush()
              
            except Exception as e:
                print(f"Error processing playlist {playlist_url}: {e}")
                continue

if __name__ == "__main__":
    presample_videos(50)
#    get_transcripts_metadata()


