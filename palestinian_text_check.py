import os
import glob

def get_processed_videos():
    with open("lev_videos_texts_processed.txt", "r") as f:
        lines = f.readlines()
        return set([line.strip() for line in lines])

if __name__ == "__main__":
    filelist = list(glob.glob("lev_videos_texts_check/*.txt"))
    processed_videos = get_processed_videos()
    non_processed_count = 0
    for file in filelist:
        video_id = file.split("/")[-1].split(".")[0]
        if video_id not in processed_videos:
            print(f"Video {video_id} not processed")
            non_processed_count += 1
            #os.remove(file)
    print(f"Total videos: {len(filelist)}")
    print(f"Non processed videos: {non_processed_count}")
