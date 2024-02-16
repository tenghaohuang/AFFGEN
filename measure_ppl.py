import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from config import Config
from GLTR import LM

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
lm = LM(model,tokenizer)

stop_words = set(stopwords.words('english'))

def get_NRC_lexicon(path):
    '''
    @output:
    - A dictionary of format {word : score}
    '''
    lexicon = path
    val_dict = {}
    aro_dict = {}
    dom_dict = {}
    with open(lexicon, 'r') as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row in reader:
            word = row['Word']
            val_dict[word] = float(row['Valence'])
            aro_dict[word] = float(row['Arousal'])
            dom_dict[word] = float(row['Dominance'])
    return (val_dict, aro_dict, dom_dict)


val_dict, aro_dict, _ = get_NRC_lexicon(Config.NRC_lexicon_path)

def remove_stop_words(text):
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_text = [i for i in filtered_text if i != "like"]
    return filtered_text

def get_arousal_score(infs):
    '''
    input:
        infs: a list of commonsense inferences
    output:
        score: the sum of valence valence scores
    '''
    if infs ==[]:
        return None,None
    sum = 0
    # print(infs)
    rt_l = []
    total_cnt = 0
    for inf in infs:
        inf = remove_stop_words(inf)

        sub_scores = []
        cnt = 0
        for part in inf:
            if part not in aro_dict:
                continue
            cnt+=1
            sub_scores.append(aro_dict[part])
        total_cnt += cnt
        if sub_scores == []:
            continue
        sum+= max(sub_scores)
        rt_l.append(max(sub_scores))
    return sum, rt_l, total_cnt

def get_clean_text(corpus):
    c_l = corpus.split(" ")
    c_l = [i.strip(".") for i in c_l]

    return c_l

def get_avg_story_score(story):
    clean_story = get_clean_text(story)
    summ, l, cnt = get_arousal_score(clean_story)
    return summ

from IPython import embed


import numpy as np
def get_ppl(story):
    lm.check_real_tok_probabilities(story)
    logits = lm.check_real_tok_probabilities(story)
    logits = [i if i > 0 else 0.00001 for i in logits]
    new_perplexity = np.prod(logits) ** (-1 / len(logits))
    return new_perplexity
