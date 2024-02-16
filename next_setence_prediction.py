from transformers import AutoModelForMultipleChoice, AutoTokenizer
import torch
import csv
from IPython import embed

prompt = "A surprise  Jack smelled something coming from the kitchen. Jack investigated the smell."
candidate1 = "A strange, strong burning sensation hit his throat."

candidate2 = "A strange, strong burning sensation hit him!"
tokenizer = AutoTokenizer.from_pretrained("hypefi/my_awesome_swag_model")
model = AutoModelForMultipleChoice.from_pretrained("hypefi/my_awesome_swag_model")


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


val_dict, aro_dict, _ = get_NRC_lexicon(
    "/nas/luka-group/tenghao/tenghaoh/creative_writing/Finetuned_GPT2/dataset/NRC-VAD-Lexicon.txt")


def get_corpus_valence_score(corpus):
    corpus = corpus.split(" ")
    words = [i.strip(".") for i in corpus]
    rt = []
    for word in words:
        score = val_dict.get(word, -1)
        if score != -1:
            rt.append(score)
    return sum(rt) / len(rt) if len(rt) != 0 else -1


def get_cloze_distribution(prompt, candidates):
    prompt_candidate_pairs = []
    for cand in candidates:
        prompt_candidate_pairs.append([prompt, cand])

    inputs = tokenizer(prompt_candidate_pairs, return_tensors="pt", padding=True)
    labels = torch.tensor(0).unsqueeze(0)

    outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
    logits = outputs.logits
    # ranks = torch.argmax(logits)
    # rt = []
    # for num, pair in enumerate(prompt_candidate_pairs):
    #     rt.append((pair, logits[0][num]))

    return logits.cpu().detach().numpy()


def select_proper_ending(prompt, candidates):
    prompt_candidate_pairs = []
    for cand in candidates:
        prompt_candidate_pairs.append([prompt, cand])

    inputs = tokenizer(prompt_candidate_pairs, return_tensors="pt", padding=True)
    labels = torch.tensor(0).unsqueeze(0)

    outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
    logits = outputs.logits

    return torch.argmax(logits)


def select_contrast_valence(prompt, candidates):
    cloze_distribution = get_cloze_distribution(prompt, candidates)
    prompt_valence = get_corpus_valence_score(prompt)
    candidates_valence = [abs(get_corpus_valence_score(i[len(prompt):]) - prompt_valence) for i in candidates]
    combined = []
    # embed()
    for num, cand in enumerate(candidates):
        combined.append((cand, cloze_distribution[0][num] / 25 + candidates_valence[num]))
    combined.sort(key=lambda x: x[1], reverse=True)
    embed()
