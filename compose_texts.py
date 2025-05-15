from langchain_core.messages import HumanMessage
import os
import time
import json
from glob import glob
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import requests
import json
import dotenv
import multiprocessing
import re

dotenv.load_dotenv()

MAX_TOKENS = 8000

MIN_TOKENS_PER_PARAGRAPH = 150

def speaker2text(speaker_id):
  sid = speaker_id.split('_')[1]
  return "** " + str(sid) + " " + "المتحدث" + " **"

def count_tokens(text_block):
  cnt_tokens = 0
  for text in text_block:
    cnt_tokens += len(text['text'].split(' '))
  return cnt_tokens

class DeepseekChat(BaseChatModel):
    api_key: str  # Deepseek API key
    model: str = "deepseek-chat"  # Default model
    temperature: float = 0.7
    max_tokens: int = 8192

    def _generate(self, messages, **kwargs):
        # Convert LangChain messages to Deepseek format
        formatted_messages = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in messages
        ]

        # Call Deepseek API
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # Extract the generated text
        content = result["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self):
        return "deepseek-chat"

model = DeepseekChat(api_key=os.getenv("DEEPSEEK_API_KEY"))

def get_transcripts_dict():
  transcripts = glob("emirati_videos_transcripts/*.json")
  return {os.path.splitext(os.path.basename(t))[0]: t for t in transcripts}

transcripts_dict = get_transcripts_dict()

def get_rttms_dict():
  rttms = glob("output/pred_rttms/*.rttm")
  return {os.path.splitext(os.path.basename(r))[0]: r for r in rttms}

rttms_dict = get_rttms_dict()

def get_json_prep(video_id):
  if video_id not in transcripts_dict or video_id not in rttms_dict:
    return None
    
  transcript_file = transcripts_dict[video_id]
  rttm_file = rttms_dict[video_id]
  
  speaker_segments = []
  with open(rttm_file, 'r') as f:
    for line in f:
      parts = line.strip().split()
      # Extract relevant info: start_time, duration, speaker_id
      start_time = float(parts[3])
      duration = float(parts[4])
      speaker_id = parts[7]
      speaker_segments.append({
          'start': start_time,
          'end': start_time + duration,
          'speaker': speaker_id
      })
  
  # Read JSON file
  with open(transcript_file, 'r') as f:
    transcript = json.load(f)
    
  enriched_transcript_6min = []
  # Add speaker_id to each text block
  for text_block in transcript:
    
    text_start = float(text_block['start'])
    
    #if text_start > 360: # 6 minutes
    #  break
      
    text_duration = float(text_block['duration'])
    text_end = text_start + text_duration
    
    # Find overlapping speaker segment
    for segment in speaker_segments:
      speaker_start = segment['start']
      speaker_end = segment['end']
        # Check if text block falls within this speaker segment
      if (text_end < speaker_start) or \
        (speaker_end < text_start):
        continue
      else:
        text_block['speaker_id'] = segment['speaker']
        break
      
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
      inner_text_block = {'text': text_block['text'], 'start': text_block['start'], 'duration': text_block['duration']}
      inner_merged_text.append(inner_text_block)
    else:
      if inner_merged_text:
        merged_transcript.append({'text': inner_merged_text, 'speaker_id': previous_speaker_id })
      inner_merged_text = [text_block]
      previous_speaker_id = text_block['speaker_id']
  if inner_merged_text:
    merged_transcript.append({'text': inner_merged_text, 'speaker_id': previous_speaker_id })
  return merged_transcript

def send_prompt(prompt):
  messages = [HumanMessage(content=prompt)]
  response = model.invoke(messages)
  return response.content

def get_processed_videos():
  with open("emirati_videos_transcripts_processed.txt", "r") as f:
    lines = f.readlines()
    return set([line.strip() for line in lines])

processed_videos = get_processed_videos()

def compose_text(video_id):
  enriched_transcript_6min = get_json_prep(video_id)
  
  if enriched_transcript_6min is None:
    print(f"No enriched transcript for {video_id}")
    return
  
  tokens_exceeded = False

  with open(f"emirati_videos_texts/{video_id}.txt", "w") as f:
    merged_transcript = merge_text_blocks_by_speaker_id(enriched_transcript_6min)
        
    for text_block in merged_transcript:
      
      cnt_tokens = count_tokens(text_block['text'])
      if cnt_tokens > MAX_TOKENS:
        tokens_exceeded = True
        break
      
      tries = 0
      
      while tries < 3:
        prompt1 = f"""Take Emirati Arabic transcript below and perform sentence segmentation with automatic punctuation restoration. Don't change the words and don't add any other text. Return the result as a plain text without any disclaimer.
      Emirati Arabic transcript:
      JSON```
      {json.dumps(text_block['text'])}
      ```"""
        text = send_prompt(prompt1)
        if re.search(r'[.!؟،؛]', text):
          break
        else:
          tries += 1
          print(f"{video_id} Sentence segmentation failed. Retrying... {tries}")
          
      text = text.replace('\n', ' ')
      
      if len(text.split(' ')) > MIN_TOKENS_PER_PARAGRAPH:
      
        prompt2 = f"""Take the Emirati Arabic text below and perform paragraph segmentation. Use double new lines to separate paragraphs. Don't change the words and don't add any other text. Return the result as a text without any disclaimer.
      Text:
      {text}"""
        tries = 0
        while tries < 3:
          time.sleep(1)
          text = send_prompt(prompt2)
          if '\n\n' in text:
            break
          else:
            tries += 1
            print(f"{video_id} Paragraph segmentation failed. Retrying... {tries}")
      f.write(speaker2text(text_block['speaker_id']) + '\n')
      f.write(text + '\n')
      f.write('\n\n\n')
      f.flush()
      time.sleep(1)
    
  if tokens_exceeded:
    print(f"Skipping {video_id} because it has more than {MAX_TOKENS} tokens")
    os.remove(f"emirati_videos_texts/{video_id}.txt")
  
  print(f"Finished processing of {video_id}")
  
if __name__ == "__main__":    
    transcript_keys = list(transcripts_dict.keys())
    transcript_keys = [key for key in transcript_keys if key not in processed_videos]
    len_transcripts = len(transcript_keys)
    chunk_size = 48
    transcript_chunks = [transcript_keys[i:i + chunk_size] for i in range(0, len(transcript_keys), chunk_size)]
    
    with multiprocessing.Pool(processes=16) as pool:
        for i, chunk in enumerate(transcript_chunks):
            print(f"Processing {i*chunk_size+1}:{i*chunk_size+len(chunk)} from {len_transcripts}") 
               
            start_time = time.time()
            
            # Process chunk in parallel
            pool.map(compose_text, chunk)
            
            end_time = time.time()
            
            print(f"Time taken: {end_time - start_time} seconds")
            
