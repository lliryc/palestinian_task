import re
import difflib
import pyarabic.araby as araby

def tokenize(text):
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return tokens

def is_punctuation(token):
    return re.fullmatch(r'[^\w\s]', token, re.UNICODE) is not None

def is_similar(word1, word2, threshold=0.75):
    word1 = araby.strip_diacritics(word1)
    word2 = araby.strip_diacritics(word2)
    ratio = difflib.SequenceMatcher(None, word1, word2).ratio()
    return ratio >= threshold

def merge_text_with_punctuation(original_text, corrected_text, max_missing_rate=0.3):
  
    orig_tokens = tokenize(original_text)
    corr_tokens = tokenize(corrected_text)

    max_missing_words = max(10, int(len(orig_tokens) * max_missing_rate))
    
    # Assume the original text consists only of words.
    orig_words = [tok for tok in orig_tokens if re.fullmatch(r'\w+', tok, re.UNICODE)]

    merged_tokens = []
    orig_index = 0  # Pointer into orig_words
    
    missing_count2 = 0

    for corr_index, corr_token in enumerate(corr_tokens):
        if is_punctuation(corr_token):
            merged_tokens.append(corr_token)
        else:
            found_index = None
            for k in range(orig_index, len(orig_words)):
                if is_similar(orig_words[k], corr_token, threshold=0.75):
                    found_index = k
                    break
            
            if found_index is not None and found_index - orig_index <= max_missing_words:
                missing_count = found_index - orig_index
                if missing_count > 0:
                    for missing in orig_words[orig_index:found_index]:
                        merged_tokens.append(missing)
                    orig_index = found_index
                merged_tokens.append(orig_words[orig_index])
                orig_index += 1
            else:
                missing_count2 +=1
                if missing_count2 > max_missing_words:
                  raise Exception(f"Too many missing words: {missing_count2} which exceeds allowed gap of {max_missing_words}.")
    
    # Append any remaining original words (if any).
    while orig_index < len(orig_words):
        merged_tokens.append(orig_words[orig_index])
        orig_index += 1
    
    # Reconstruct the merged text.
    merged_text = ""
    for i, token in enumerate(merged_tokens):
        if i == 0:
            merged_text += token
        else:
            if is_punctuation(token):
                merged_text += token
            else:
                merged_text += " " + token

    return merged_text

# --- Example usage with debug output ---
if __name__ == "__main__":
    original_text = """
، ‏p مرحبا واهلا وسهلا فيكم بحلقه جديده ، وغنيه باكله مع بانوس اليوم معنا تعودت ، عليها عيده اهلا وسهلا بطل بدي اقول لك . اهلا وسهلا هيدا صار محلك كمان ا اليوم ، انا ما ح اقول لكم الجريدي لانه الصراحه ، بتصير ذوقوا وشموا لكم مش ضابطه عيدا انا ، اللي شايفته شايفه نوري وشايفه سومو ورادس ، و والسيكس تبع السوشي بعد انه سوشي بس ه . يعني شايفه عده انواع مي تفضلي قوليلنا 
"""
    corrected_text = """
مرحبا واهلا وسهلا فيكم بحلقة جديدة، وغنية بأكلة مع بانوس. اليوم معنا تعودت عليها عيدة. اهلا وسهلا بطل بدي اقول لك اهلا وسهلا، هيدا صار محلك كمان. اليوم انا ما ح اقول لكم الجريدي لانه الصراحة بتصير "ذوقوا وشموا" لكم مش ضابطة عيدة. انا اللي شايفته شايفة نوري وشايفة سومو ورادس والسيكس تبع السوشي، بعد انه سوشي بس. يعني شايفة عدة انواع، مي تفضلي قوليلنا.
"""
    
    try:
        merged_text = merge_text_with_punctuation(original_text, corrected_text)
        print(merged_text)
    except Exception as e:
        print("Exception raised:", e)