import csv

from utils import remove_stop_words, finish_story, chunks, get_GPT2_sampling_mode
import scipy.stats as ss


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


val_dict, aro_dict, _ = get_NRC_lexicon("../Finetuned_GPT2/dataset/NRC-VAD-Lexicon.txt")


def get_arousal_score(infs):
    '''
    input:
        infs: a list of commonsense inferences
    output:
        score: the sum of valence valence scores
    '''
    if infs == []:
        return None, None
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
            cnt += 1
            sub_scores.append(aro_dict[part])
        total_cnt += cnt
        if sub_scores == []:
            continue
        sum += max(sub_scores)
        rt_l.append(max(sub_scores))
    return sum, rt_l, total_cnt


def get_corpus_valence_score(corpus):
    corpus = corpus.split(" ")
    words = [i.strip(".") for i in corpus]
    rt = []
    for word in words:
        score = val_dict.get(word, -1)
        if score != -1:
            rt.append(score)
    return sum(rt) / len(rt) if len(rt) != 0 else -1


def get_valence_score(infs):
    '''
    input:
        infs: a list of commonsense inferences
    output:
        score: the sum of valence valence scores
    '''
    if infs == []:
        return None, None
    sum = 0

    rt_l = []
    for inf in infs:
        inf = remove_stop_words(inf)

        sub_scores = []
        cnt = 0
        for part in inf:
            if part not in val_dict:
                continue
            cnt += 1
            sub_scores.append(val_dict[part])
        if sub_scores == []:
            continue
        sum += max(sub_scores)
        rt_l.append(max(sub_scores))
    return sum, rt_l


def get_most_opp_valence(args, first_stage, second_stage, model, tokenizer):
    most_oppo_val_stories = []
    for setting_gene, cont_genes in zip(first_stage, chunks(second_stage, args.rounds_of_generation)):
        setting_inf_score = get_valence_score(setting_gene[args.surprise_position - 1]['<|xReact|>'])
        cont_gap_scores = []
        for cont_gene in cont_genes:
            # print(cont_gene)
            cont_inf_score = get_valence_score(cont_gene[args.surprise_position]['<|xReact|>'])
            cont_gap_scores.append(abs(cont_inf_score - setting_inf_score))
        prompt = cont_genes[cont_gap_scores.index(max(cont_gap_scores))]['story']
        _, full_story = finish_story(prompt, model, tokenizer, num_story_return=1)
        most_oppo_val_stories.append(full_story)
    return most_oppo_val_stories
