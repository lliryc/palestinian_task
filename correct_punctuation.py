from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import random

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

prompt = ChatPromptTemplate.from_template("""You are Levantine Arabic speaker. Your task is to correct the punctuation in the Levantine Arabic text below. Return only the corrected text without any other explanation.

{text}""")

# Anthropic model
anthropic_model = 'claude-3-5-sonnet-20241022'
    
llm_anthropic = ChatAnthropic(anthropic_api_key=ANTHROPIC_API_KEY, model=anthropic_model)

chain_anthropic = prompt | llm_anthropic


def correct_punctuation(text):
    try:
        res = chain_anthropic.invoke({"text": text})
    except Exception as e:
        print(e)
        time.sleep(random.randint(2, 5))
        return None
   
    return res.content


if __name__ == "__main__":

    for file in tqdm(glob.glob("lev_videos_texts_check/*.txt")):
        file_name = file.split("/")[-1]
        processed = False
        
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        with open(f"lev_videos_texts_punct_corrected/{file_name}", "w", encoding="utf-8") as f:
            speakers_blocks = raw_text.split("\n\n\n\n")
            for speaker_block in speakers_blocks:
                speaker_line = speaker_block.split("\n")[0]
                f.write(speaker_line + "\n")
                speaker_text = speaker_block.replace(speaker_line + "\n", "").strip()
                paragraphs = speaker_text.split("\n\n")
                for paragraph in paragraphs:
                    if paragraph.strip() == "":
                        continue                                        
                    corrected_paragraph = correct_punctuation(paragraph)                    
                    if corrected_paragraph:
                        corrected_paragraph = corrected_paragraph.split("\n\n")[0]                    
                        f.write(corrected_paragraph + "\n\n")
                        processed = True
                    else:
                        processed = False
                        break
                if not processed:
                    break
        
        if not processed:
            print(f"Failed to process {file_name}")
            os.remove(f"lev_videos_texts_punct_corrected/{file_name}")

