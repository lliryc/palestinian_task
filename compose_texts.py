from langchain_core.messages import HumanMessage
import os
import time
import json
from glob import glob
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import requests
import json
import dotenv
import multiprocessing
import numpy as np
import re
import math
from glob import glob

dotenv.load_dotenv()

MAX_TOKENS = 8000

MIN_TOKENS_PER_PARAGRAPH = 150

api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="o3-mini", api_key=api_key)

def speaker2text(speaker_id):
  sid = speaker_id.split('_')[1]
  return "** " + str(sid) + " " + "المتحدث" + " **"

def speaker2text2_time_period(speaker_id, start, end):
  sid = speaker_id.split('_')[1]
  return "[" + str(start) + " - " + str(end) + "] " + " ** " + str(sid) + " " + "المتحدث" + " **"

def count_tokens(text_block):
  cnt_tokens = 0
  for text in text_block:
    cnt_tokens += len(text["text"].split(" "))
  return cnt_tokens

api_key = os.getenv("OPENAI_API_KEY")

#model = ChatOpenAI(model="o1-mini", api_key=api_key)

def get_transcripts_dict():

  transcripts = glob("palestinian_videos_transcripts/*.json")
  return {os.path.splitext(os.path.basename(t))[0]: t for t in transcripts}

transcripts_dict = get_transcripts_dict()

def get_rttms_dict():
  rttms = glob("output/pred_rttms/*.rttm")
  return {os.path.splitext(os.path.basename(r))[0]: r for r in rttms}

rttms_dict = get_rttms_dict()

def transcripts_prep(transcript_json, end_time):
  new_transcript_json = []
  len_transcript = len(transcript_json)
  for i, _ in enumerate(transcript_json):
    text_start = float(transcript_json[i]['start'])
    if text_start > end_time:
      break
    duration = float(transcript_json[i]['duration'])
    next_text_start = transcript_json[i + 1]['start'] - 0.1 if i < len_transcript - 1 else text_start + duration    
    text_end = min(text_start + duration, next_text_start)
    duration = min(duration, next_text_start - text_start - 0.1)
    new_transcript_json.append({'text': transcript_json[i]['text'], 'start': text_start, 'end': text_end, 'duration': transcript_json[i]['duration']})
  return new_transcript_json
      

def get_json_prep(video_id):
  if video_id not in transcripts_dict or video_id not in rttms_dict:
    return None
    
  transcript_file = transcripts_dict[video_id]
  rttm_file = rttms_dict[video_id]
  
  speaker_segments = []

  with open(rttm_file, 'r') as f:
    lines = f.readlines()
    len_lines = len(lines)
    for i, _ in enumerate(lines):
      parts = lines[i].strip().split()    
      # Extract relevant info: start_time, duration, speaker_id
      start_time = float(parts[3])
      duration = float(parts[4])

      if i < len_lines - 1:
        next_start_time = float(lines[i + 1].strip().split()[3]) - 0.1
        end_time = min(start_time + duration, next_start_time)
        duration = min(duration, next_start_time - start_time - 0.1)
      else:
        end_time = start_time + duration

      speaker_id = parts[7]
      speaker_segments.append({
          'start': start_time,
          'end': end_time,
          'speaker': speaker_id
      })
  
  # Read JSON file
  with open(transcript_file, 'r') as f:
    transcript = json.load(f)

  transformed_transcript_6min = transcripts_prep(transcript, 360)
    
  enriched_transcript_6min = []
  # Add speaker_id to each text block
  for text_block in transformed_transcript_6min:
    
    text_start = float(text_block['start'])          
    text_end = float(text_block['end'])
    intersected_speaker_segments = []
    intersected_speaker_segments_lengths = []
    # Find overlapping speaker segment
    for segment in speaker_segments:
      speaker_start = segment['start']
      speaker_end = segment['end']
        # Check if text block falls within this speaker segment
      if (text_end < speaker_start) or \
        (speaker_end < text_start):
        if len(intersected_speaker_segments) > 0:
          ix = np.argmax(intersected_speaker_segments_lengths)
          text_block['speaker_id'] = intersected_speaker_segments[ix]
          intersected_speaker_segments_lengths = []
          intersected_speaker_segments = []
          break
        else:
          continue
      else:
        intersected_speaker_segments.append(segment['speaker'])
        intersected_speaker_segments_lengths.append(min(speaker_end, text_end) - max(speaker_start, text_start))
    
    if len(intersected_speaker_segments) > 0 and 'speaker_id' not in text_block:
      ix = np.argmax(intersected_speaker_segments_lengths)
      text_block['speaker_id'] = intersected_speaker_segments[ix]
      intersected_speaker_segments_lengths = []
      intersected_speaker_segments = []

      
    if 'speaker_id' not in text_block:
      continue
    
    enriched_transcript_6min.append(text_block)
  
  return enriched_transcript_6min

def merge_text_blocks_by_speaker_id(json_transcript):

  previous_speaker_id = None
  merged_transcript = []
  inner_merged_text = []
  
  for text_block in json_transcript:

    if text_block['speaker_id'] == previous_speaker_id:

      inner_text_block = {'text': text_block['text'], 'start': text_block['start'], 'duration': text_block['duration'], 'end': text_block['end']}
      inner_merged_text.append(inner_text_block)
    else:

      if inner_merged_text:
        merged_transcript.append({'text': inner_merged_text, 'speaker_id': previous_speaker_id })      
      
      previous_speaker_id = text_block['speaker_id']
      inner_merged_text = [{'text': text_block['text'], 'start': text_block['start'], 'duration': text_block['duration'], 'end': text_block['end']}]
  
  if inner_merged_text:
    merged_transcript.append({'text': inner_merged_text, 'speaker_id': previous_speaker_id })
  
  return merged_transcript

#def send_prompt(prompt):
#  messages = [HumanMessage(content=prompt)]
#  response = model.invoke(messages)
#  return response.content

def get_processed_videos():
  with open("palestinian_videos_transcripts_processed.txt", "r") as f:
    lines = f.readlines()
    return set([line.strip() for line in lines])

def get_file_ids():
  files = glob("palestinian_videos_texts/*.*")
  file_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
  return list(set(file_ids))

processed_videos = get_file_ids()

def write_processed_videos(video_ids):
  with open("palestinian_videos_transcripts_processed.txt", "a") as f:
    for video_id in video_ids:
      f.write(video_id + "\n")

def adjust_speech_ids(text_list):
  new_text_list = []
  id = 0
  for text_item in text_list:
    new_text_item = {'text': text_item['text'], 'start': text_item['start'], 'end': text_item['end'], 'id': id}
    new_text_list.append(new_text_item)  
    id += 1
  return new_text_list

segmentation_output_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Palestinian Arabic Transcript Segmentation",
  "description": "A list of segmentation markers for Palestinian Arabic transcripts. Each object indicates punctuation mark is inserted next to text item with the given id",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "integer",
        "description": "Identifier of the text item in the input transcript"
      },
      "punctuation": {
        "type": "string",
        "enum": ["،", "؛", "؟", "…", "...", ".", "!", ":", ""],
        "description": "Allowed Arabic punctuation mark to be inserted next to text item"
      },
    },
    "required": ["id", "punctuation"],
    "additionalProperties": False
  }
}

def send_prompt(prompt):
  messages = [HumanMessage(content=prompt)]
  response = model.invoke(messages)
  return response.content

def extract_text_from_textlist(text_list, segmentation_output):
  dict_segments = {(str(item['id'])): item for item in segmentation_output}
  text = ""
  
  for text_item in text_list:  
    text_id = str(text_item['id'])
    if text_id in dict_segments:
      punctuation = dict_segments[text_id]['punctuation']
      formatted_phrase = punctuation + " " + text_item['text']   
      text += formatted_phrase + " "
    else:
      text += text_item['text'] + " "
    
    
  return text

def compose_text(video_id):
  enriched_transcript_6min = get_json_prep(video_id)
  
  if enriched_transcript_6min is None:
    print(f"No enriched transcript for {video_id}")
    return
  
  tokens_exceeded = False

  with open(f"palestinian_videos_texts/{video_id}.txt", "w") as f:

    merged_transcript = merge_text_blocks_by_speaker_id(enriched_transcript_6min)
        
    for text_block in merged_transcript:
      
      ttext_list = adjust_speech_ids(text_block['text'])
      
      cnt_tokens = count_tokens(ttext_list)
      if cnt_tokens > MAX_TOKENS:
        tokens_exceeded = True
        break

      tries = 0
      segmentation_output = None
      
      while tries < 10:
        prompt1 = f"""Take Palestinian Arabic transcript in a JSON format and add punctuation marks between text items if needed by analyzing the text as well as start and end times. Return result in a JSON format according to output schema without any disclaimer.

      Output schema:
      JSON```
      {json.dumps(segmentation_output_schema)}
      ```

      Palestinian Arabic transcript:
      JSON```
      {json.dumps(ttext_list)}
      ```"""        
        try:
          output = send_prompt(prompt1)
          output = output.replace('```json', '').replace('```', '').strip()
          segmentation_output = json.loads(output)
          # Validate segmentation_output is a list
          if not isinstance(segmentation_output, list):
            raise ValueError("Segmentation output must be a list")
          # Validate each item in the list
          for item in segmentation_output:
            if not isinstance(item, dict):
              raise ValueError("Each item in segmentation output must be a dictionary")
            if 'id' not in item:
              raise ValueError("Missing required fields in segmentation output")
          break
        except Exception as e:
          tries += 1
          print(f"{video_id} Sentence segmentation failed. Retrying... {tries}")
          print(f"Error: {str(e)}")
          time.sleep(1)  # Add small delay between retries
          
      text = extract_text_from_textlist(ttext_list, segmentation_output)
      start = math.floor(ttext_list[0]['start'])
      end = math.ceil(ttext_list[-1]['end'])
      f.write(speaker2text2_time_period(text_block['speaker_id'], start, end) + '\n')
      f.write(text + '\n')
      f.write('\n\n\n')
      f.flush()
      time.sleep(1)
    
  if tokens_exceeded:
    print(f"Skipping {video_id} because it has more than {MAX_TOKENS} tokens")
    os.remove(f"palestinian_videos_texts/{video_id}.txt")
  
  print(f"Finished processing of {video_id}")

def get_text_from_textblock(text_block):
  text = ""
  for text_item in text_block['text']:
    text += text_item['text'] + " "
  return text

def compose_text2(video_id):
  enriched_transcript_6min = get_json_prep(video_id)
  
  if enriched_transcript_6min is None:
    print(f"No enriched transcript for {video_id}")
    return
  
  with open(f"palestinian_videos_texts/{video_id}.txt", "w") as f:
    merged_transcript = merge_text_blocks_by_speaker_id(enriched_transcript_6min)
        
    for text_block in merged_transcript:      
           
      f.write(speaker2text(text_block['speaker_id']) + '\n')
      f.write(get_text_from_textblock(text_block) + '\n')
      f.write('\n\n\n')
      f.flush()
      time.sleep(1)    
  
  print(f"Finished processing of {video_id}")

  
if __name__ == "__main__":
        
    transcript_keys = list(transcripts_dict.keys())
    
    transcript_keys = [key for key in transcript_keys if key not in processed_videos]
    
    len_transcripts = len(transcript_keys)
    
    chunk_size = 48
    
    transcript_chunks = [transcript_keys[i:i + chunk_size] for i in range(0, len(transcript_keys), chunk_size)]

    with multiprocessing.Pool(processes=16) as pool:

      for i, chunk in enumerate(transcript_chunks):

        print(f"Processing {i * chunk_size + 1}:{i * chunk_size + len(chunk)} from {len_transcripts}")              
        
        start_time = time.time()
        # Process chunk in parallel
        pool.map(compose_text, chunk)

        end_time = time.time()
        
        print(f"Time taken: {end_time - start_time} seconds")
        
        write_processed_videos(chunk)
