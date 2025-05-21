import re
import difflib
import pyarabic.araby as araby

class MergeException(Exception):
    pass

def tokenize(text):
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return tokens

def tokenize_processed(text):

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    all_tokens = []
    para_start_indices = []
    token_count = 0
    for para in paragraphs:
        para_start_indices.append(token_count)
        tokens = tokenize(para)
        all_tokens.extend(tokens)
        token_count += len(tokens)
    return all_tokens, para_start_indices

def tokenize_with_positions(text):

    pattern = re.compile(r'\w+|[^\w\s]', re.UNICODE)
    tokens = []
    for match in pattern.finditer(text):
        token = match.group(0)
        start = match.start()
        end = match.end()
        tokens.append((token, start, end))
    return tokens

def merge_text_with_paragraphs(original_text, processed_text, diff_threshold=0.65):

    orig_tokens_with_pos = tokenize_with_positions(original_text)
    orig_token_list = [token for token, start, end in orig_tokens_with_pos]
    
    proc_tokens, para_start_indices = tokenize_processed(processed_text)
    
    proc_tokens_dediac = [araby.strip_diacritics(token) for token in proc_tokens]
    orig_token_list_dediac = [araby.strip_diacritics(token) for token in orig_token_list]

    
    # 2. Align tokens using difflib.
    matcher = difflib.SequenceMatcher(a=orig_token_list_dediac, b=proc_tokens_dediac)
    similarity = matcher.ratio()
    if similarity < diff_threshold:
        raise MergeException(
            f"Difference threshold exceeded: similarity {similarity:.2f} is below {diff_threshold}."
        )
    
    proc_to_orig_mapping = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for offset in range(j2 - j1):
                proc_index = j1 + offset
                orig_index = i1 + offset
                proc_to_orig_mapping[proc_index] = orig_index

    insertion_positions = []
    for para_start in para_start_indices[1:]:
        # If the first token of this paragraph exists in the mapping, get its original token index.
        if para_start in proc_to_orig_mapping:
            orig_index = proc_to_orig_mapping[para_start]
            # Insert break before this token in the original text:
            token, token_start, token_end = orig_tokens_with_pos[orig_index]
            insertion_positions.append(token_start)
        else:
            candidates = [proc_to_orig_mapping[idx] for idx in proc_to_orig_mapping if idx < para_start]
            if candidates:
                orig_index = max(candidates)
                _, _, token_end = orig_tokens_with_pos[orig_index]
                insertion_positions.append(token_end)
    
    merged_text = original_text
    for pos in sorted(insertion_positions, reverse=True):
        merged_text = merged_text[:pos] + "\n" + merged_text[pos:]
    
    return merged_text

# ==========================
# Example usage with your texts:
# ==========================
if __name__ == "__main__":
    original = """الطريق الى الماعز في بنجلادش يمر من هنا. خليني افرجيكم كيف الناس بتعيش في القرى. كيوشو انت جايني ط تعرف انا وين عنزه جاي معي. رحله فانت تاكل رايس من اللي انت تزرعه. اليوم الصبح سالت حالي شو احسن هديه تاخذها اذا بدك تزور قريه في ريف بنجلادش، خصوصا انه هلا وقت عيد؟ هل اخذ صفط كاسات قهوه ولا اخذ شوكولاته ولا معمول، بس بالاخر قررت اشتري معزه. الهدف اليوم اني اروح على احد القرى في بنجلادش اشوف كيف بعيشوا وكيف راح يعيدوا، لانه بكره اول ايام عيد الفطر. بالبيت عننا في الاردن في بنت بتيجي كل اسبوع تساعد امي في شغل البيت، يعني البنت ها اسمها فاطمه من بنجلادش. فاعطتني شويه اغراض وصلها ليلتها عشان ينبسطوا: في شوكولاته، في ميه زمزم، في كانه اكسسوارات لبنتها. بعد شوي راح يجي اخوها ياخذني ونروح لعندهم، اتوقع هم عايشين برا. دكه خلينا نروح نزورهم نشوف كيف بعيشوا هناك. السلام عليكم، كيف الحال؟- السلام، الحمد لله.- حبيبي انت فين؟ انت قلت ثلث ساعه، هذه ساعه ونص!- والله يا زياي يجي قريب قريب قريب.- موجود قريب يعني كم وقت كمان؟- ثلث ساعه.- لا! لا! لا! خم دقيقه.- اي اشاره يا اخي؟- ما فيش اشاره في البلد كلها. انت عندي اي اشاره؟- دوار الداخليه.- انت جايني؟ طب تعرف انا وين؟- ما تجيني يا الله، انا بستسلم بستسلم.- في دوار، دوار. انت يعرف فين الدوار؟- انت عندك واتساب؟ طب انا يرسل لوكيشن انت ما يجي؟- ليش كل ما برن عليه بوللي جايلك؟ طب جايلي وين؟ مش عارف وين انا!- الساعه صارت اعه العصر، مفروض نلتقي على الوحده. نا ساعه بس بنلف بنفس المكان!- سيلفي، ا سيلفي.- يلا اوكي اوكي.- ويلكم، بيبسي انت تشرب بيبسي؟- ري س ريد سي! يس يس يسم!"""

    processed = """لطريق الى الماعز في بنجلادش يمر من هنا. خليني افرجيكم كيف الناس بتعيش في القرى. كيوشو انت جايني ط تعرف انا وين عنزه جاي معي. رحله فانت تاكل رايس من اللي انت تزرعه.
    لطريق الى الماعز في بنجلادش يمر من هنا. خليني افرجيكم كيف الناس بتعيش في القرى. كيوشو انت جايني ط تعرف انا وين عنزه جاي معي. رحله فانت تاكل رايس من اللي انت تزرعه.
    لطريق الى الماعز في بنجلادش يمر من هنا. خليني افرجيكم كيف الناس بتعيش في القرى. كيوشو انت جايني ط تعرف انا وين عنزه جاي معي. رحله فانت تاكل رايس من اللي انت تزرعه.
    لطريق الى الماعز في بنجلادش يمر من هنا. خليني افرجيكم كيف الناس بتعيش في القرى. كيوشو انت جايني ط تعرف انا وين عنزه جاي معي. رحله فانت تاكل رايس من اللي انت تزرعه.
    """

    try:
        merged = merge_text_with_paragraphs(original, processed, diff_threshold=0.7)
        print("Merged Text:\n", merged)
    except MergeException as me:
        print("Merge Exception:", me)
