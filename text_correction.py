from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import random
import punctuation
import paragraphs
import pyarabic.araby as araby
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

punctuation_prompt = ChatPromptTemplate.from_template("""You are Syrian Arabic speaker. Your task is to correct the punctuation in the Syrian Arabic text below. Don't change words, just correct the punctuation. Return only the corrected text without any other explanation.

{text}""")

# Anthropic model
llm_openai = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY)


def run_prompt(prompt, text):
    chain = prompt | llm_openai
    try:
        res = chain.invoke({"text": text})  
    except Exception as e:
        print(e)
        time.sleep(random.randint(2, 5))
        return None
   
    return res.content


MIN_PARAGRAPH_TOKENS = 150

def paragraph_required(text):
    """Check if text needs paragraph formatting by looking for newlines and bullet points"""
    
    # Check for bullet points or dashes
    if re.search(r'[-•]', text):
        return True
        
    # Check for empty lines (original logic)
    text = text.split(" ")
    if len(text) <= MIN_PARAGRAPH_TOKENS:
        return False    
    return True

paragraphs_prompt = ChatPromptTemplate.from_template("""You are Syrian Arabic speaker. Your task is to split Syrian Arabic text into paragraphs if it makes sense. Otherwise, return the original text. Don't change words and punctuation. Return only the corrected text without any disclaimers.

{text}""")


if __name__ == "__main__":

    for file in tqdm(glob.glob("syrian_videos_texts_check/*.txt")):
        file_name = file.split("/")[-1]
        processed = False
        
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        with open(f"syrian_videos_texts_punct_corrected/{file_name}", "w", encoding="utf-8") as f:
            speakers_blocks = raw_text.split("\n\n\n\n")
            for speaker_block in speakers_blocks:
                if speaker_block.strip() == "":
                    continue                
                speaker_line = speaker_block.split("\n")[0]
                f.write(speaker_line + "\n")
                speaker_text = speaker_block.replace(speaker_line + "\n", "").strip()
                speaker_text = speaker_text.replace("\n", " ")
                
                if speaker_text.strip() == "":
                    processed = False
                    break

                tries = 0
                while tries < 5:
                    try:
                        distorted_text = run_prompt(punctuation_prompt, speaker_text)
                        distorted_text = distorted_text.replace("\n", " ")
                        merged_text = punctuation.merge_text_with_punctuation(speaker_text, distorted_text)
                        break                        
                    except Exception as e:
                        print(f"Error in punctuation correction: {e} for {file_name}. Next try...{tries+1}")
                        time.sleep(random.randint(2, 5))
                        tries += 1
                
                if not merged_text:
                    processed = False
                    break

                if paragraph_required(merged_text):
                    tries = 0
                    while tries < 5:
                        try:
                            corrected_text = run_prompt(paragraphs_prompt, merged_text)
                            merged_text = paragraphs.merge_text_with_paragraphs(merged_text,corrected_text)
                            break
                        except Exception as e:
                            print(f"Error in paragraphs correction: {e} for {file_name}. Next try...{tries+1}")
                            time.sleep(random.randint(2, 5))
                            tries += 1
                
                if merged_text:
                    f.write(merged_text + "\n\n")
                    processed = True
                
                else:
                    processed = False
                    break
        
        if not processed:
            print(f"Failed to process {file_name}")
            os.remove(f"syrian_videos_texts_punct_corrected/{file_name}")



