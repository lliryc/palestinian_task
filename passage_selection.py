from langchain_core.messages import HumanMessage
import os
import time
import json
import glob
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import requests
import json
import dotenv
import multiprocessing
import random
import re
from bisect import bisect_left, bisect_right

load_dotenv()

#ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

prompt = ChatPromptTemplate.from_template("""You are professional linguist. You are helping me with following:
Extract passage from a source text in Syrian Arabic for reading comprehension task. 
### Target audience: Syrian Arabic speakers.
### Passage selection criteria: 
- Select a consistent part of a source text suitable for inference and analytical questions
- Must consist of complete sentences only (no partial sentences at start or end)
- Passage size is about 300-500 words
- Strictly avoid any sections containing:
  * Subscription requests
  * Social media follow requests
  * Like/share/subscribe calls to action
  * Promotional content or advertisements
  * Channel/content marketing messages
### Output format: JSON with fields without any additional text: start position (int, from source text, character-wise), end position (int, character-wise). I.e. {{"start": 100, "end": 200}}
### Source text:
{text}""")

# Anthropic model
llm_openai = ChatOpenAI(api_key=OPENAI_API_KEY, model="o3-mini")

chain_openai = prompt | llm_openai

dotenv.load_dotenv()

MAX_TOKENS = 8000

def count_tokens(text):
    return len(text.split(' '))

def extract(text):
    try:
        res = chain_openai.invoke({"text": text})
    except Exception as e:
        print(e)
        return None
    res_text = res.content
    res_text = res_text.replace("```json", "").replace("```", "")
    return res_text


def get_processed_passages():
  with open("syrian_processed_passages.txt", "r") as f:
    lines = f.readlines()
    return set([line.strip() for line in lines])

def get_speaker_indices(text):
  matches = re.finditer(r'\[\d+', text)
  positions = [match.start() for match in matches]
  return positions

def get_new_line_indices(text):
  matches = re.finditer(r'\n', text)
  positions = [match.start() for match in matches]
  return positions

def get_punkt_indices(text):
    pattern = r'[،؟!.]|\.{3}|\n'  # Matches ، or ؟ or ! or . or ...
    matches = re.finditer(pattern, text)
    positions = [match.start() for match in matches]
    return positions

processed_passages = get_processed_passages()

def add_processed_passages(files):
  with open("syrian_processed_passages.txt", "a") as f:
    for file in files:
      f.write(file + "\n")

def preprocess_text(text):
  text = text.replace("\n\n\n\n", "\n\n")
  #text = re.sub(r'\[[^\]]*\]', ' ', text)
  return text

def align_start(start, punkt_indices, speaker_indices):
   if start < 0:
     return 0
   start_index1 = bisect_left(punkt_indices, start)
   if start_index1 == 0:
    punkt_pos = 0
   else:
     punkt_pos = punkt_indices[start_index1-1] + 1
   start_index2 = bisect_left(speaker_indices, start)
   if start_index2 == 0:
     speaker_pos = 0
   else:
     speaker_pos = speaker_indices[start_index2-1]
   return punkt_pos if punkt_pos > speaker_pos else speaker_pos

def align_end(end, punkt_indices, speaker_indices, max_len):
   if end >= max_len:
     return max_len - 1
   end_index1 = bisect_right(punkt_indices, end)
   if end_index1 == len(punkt_indices):
     punkt_pos = max_len - 1
   else:
     punkt_pos = punkt_indices[end_index1]
   end_index2 = bisect_right(speaker_indices, end)
   if end_index2 == len(speaker_indices):
     speaker_pos = max_len - 1
   else:
     speaker_pos = speaker_indices[end_index2] - 1
   return punkt_pos if punkt_pos<speaker_pos else speaker_pos

def get_speaker(speaker_indices, start, text):
  start_index = bisect_left(speaker_indices, start)
  
  if start_index == 0:
    return 0
  
  else:
    start_speaker = speaker_indices[start_index - 1]
    pattern = r'^\[(\d+) - (\d+)\]\s+\*\*\s+(\d+)\s+[\u0600-\u06FF\s]+\*\*'
    matches = re.finditer(pattern, text[start_speaker:start_speaker+100])
    for match in matches:
      if match.start() == 0:
        return match.group(0)
    return None

def extract_passage(source_text_file):
  filename = source_text_file.split("/")[-1]

  with open(source_text_file, "r") as f:
      text = f.read()

  text = preprocess_text(text)

  cnt_tokens = count_tokens(text)

  if cnt_tokens < 200:
    print(f"{filename} : Original text is too short: {cnt_tokens}")
    return False
  
  if cnt_tokens <= 600:
    output_file = f"syrian_passages/{filename}"
    with open(output_file, "w") as f:
      f.write(text)
    print(f"{output_file} is saved")
    return True
  
  speaker_indices = get_speaker_indices(text)
  punkt_indices = get_punkt_indices(text)
  
  tries = 0

  passage = None
  
  while tries < 3:
    res_text = extract(text)
    try:
      json_response = json.loads(res_text)
      start = int(json_response['start'])
      start = align_start(start, punkt_indices, speaker_indices)
      end = int(json_response['end'])
      end = align_end(end, punkt_indices, speaker_indices, len(text))
      if end >= len(text):
        end = len(text) - 1
      if start < 0 or end < 0 or start > end:
        raise Exception(f"Invalid start or end position ({start}, {end}) from (0, {len(text)})")
      test_passage = text[start:(end+1)]
      cnt_tokens = count_tokens(test_passage)
      if cnt_tokens < 200 or cnt_tokens > 650:
        raise Exception(f"Invalid passage size: {cnt_tokens}")
      
      if not re.match(r'^\[\d+', test_passage):
        speaker_label = get_speaker(speaker_indices, start, text)
        test_passage = speaker_label + "\n" + test_passage.lstrip()
      passage = test_passage
      break

    except Exception as e:
      print(e)
      print(f"Error extracting passage from {filename}: Retrying... {tries+1}")
      tries += 1
      time.sleep(random.randint(1, 2))
      continue
    
  if passage is not None:
    output_file = f"syrian_passages/{filename}"
    with open(output_file, "w") as f:
      f.write(passage)
    print(f"{output_file} is saved")

  return passage is not None

if __name__ == "__main__":    
    input_files = glob.glob("syrian_videos_texts2/*.txt")
    processed = get_processed_passages()
    input_files = [file for file in input_files if file not in processed]

    chunk_size = 48
    transcript_chunks = [input_files[i:i + chunk_size] for i in range(0, len(input_files), chunk_size)]
    
    #with multiprocessing.Pool(processes=1) as pool:
    for i, chunk in enumerate(transcript_chunks):
        print(f"Processing {i*chunk_size+1}:{i*chunk_size+len(chunk)} from {len(input_files)}") 
            
        start_time = time.time()
        
        processed_passages = []

        # Process chunk in parallel
        with multiprocessing.Pool(processes=16) as pool:
          results = pool.map(extract_passage, chunk)    
        #for l in range(len(chunk)):
        #  if extract_passage(chunk[l]):
        #      processed_passages.append(chunk[l])
        for file, result in zip(chunk, results):
            if result:
                processed_passages.append(file)

        add_processed_passages(processed_passages)
        
        end_time = time.time()
        
        print(f"Time taken: {end_time - start_time} seconds")