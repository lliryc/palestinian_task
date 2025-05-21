import os
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from glob import glob
import subprocess


config_path = "diar_infer_general.yaml"

# Load config
cfg = OmegaConf.load(config_path)

# Set required parameters
cfg.diarizer.manifest_filepath = "input_manifest.json"  # Replace with your manifest file path
cfg.diarizer.out_dir = "output"  # Replace with your desired output directory

def write_processed_file(file):
    with open("processed_files.txt", 'a') as f:
        f.write(file + "\n")
        f.flush()

def get_processed_files():
    with open("processed_files.txt", 'r') as f:
        processed_files = f.readlines()
    processed_files = [file.strip() for file in processed_files]
    return set(processed_files)

def diarize():
    files = list(glob('palestinian_videos/*.*'))
    f_len = len(files)
    processed_files = get_processed_files()
    for i, file in enumerate(files):
        if file in processed_files:
            print(f"Skipping {file} because it has already been processed")
            continue
        print(f"Processing file {i+1} of {f_len}")
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(file))[0]
        wav_file = f"{base_name}.wav"
        
        # Convert webm to wav using ffmpeg
        ffmpeg_command = [
            'ffmpeg', '-i', file,
            '-vn',                  # No video
            '-acodec', 'pcm_s16le', # Audio codec
            '-ar', '16000',         # Sample rate
            '-ac', '1',             # Mono channel
            wav_file
        ]

        try:
            result = subprocess.run(ffmpeg_command)
            if result.returncode != 0:
                print(f"Error converting {file} to {wav_file}: {result.returncode}")
                write_processed_file(file)
                continue
            with open("input_manifest.json", 'w') as f:
                f.write(f'{{"audio_filepath": "{wav_file}", "offset": 0, "label": "infer", "text": "-"}}\n')

            diarizer = ClusteringDiarizer(cfg=cfg)
            diarizer.diarize()
            write_processed_file(file)
        except Exception as e:
            print(f"Error converting {file} to {wav_file}: {e}")
            write_processed_file(file)       
        
        os.remove(wav_file)


diarize()
