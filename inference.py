import math
from nltk.tokenize import sent_tokenize, word_tokenize
from IPython import embed
from valence_measure import get_valence_score, get_arousal_score
import pickle
from tqdm import tqdm
import DataGenerator
from config import Config
from utils import wrapup_input, finish_story
import openai
from next_setence_prediction import select_proper_ending, get_cloze_distribution, get_corpus_valence_score
import random

openai.api_key = ""

puncts = [".", "!", "?"]
arms = 3
features = 5
rewardType = 'positive'
# rewardType='binary'
featureType = 'integer'

dg = DataGenerator.DataGenerator(arms, features, feature_type=featureType, reward_type=rewardType)

policy_maker_path = Config.policy_maker_path
positiveStrategy, simulator, records, total_regret = pickle.load(open(policy_maker_path, 'rb'))
print(policy_maker_path)
def check_puncts(txt):
    cnt = 0
    clean = ""
    for s in txt:
        if s in puncts:
            cnt +=1
        else:
            clean+=s
    return cnt,clean

def length_penalty(gama, alpha=1.5):
    return (5+gama)**alpha/6**alpha

def test_mode(prompt, stop_at, positiveStrategy=None):

    prompts_ppls = [[prompt,1]]
    crt_length = 0
    crt_budget = 10000
    cnt = 0
    finished = []
    minimum_len = 6
    collection = []
    decision_path = []
    initial_num, clean_txt = check_puncts(prompt)
    while True:
        crt_length = crt_length + 1
        txts_at_timestep, overall_features, overall_rewards, buckets = dg.generate_features_rewards\
            (prompts_ppls,crt_length, crt_budget, mode="test",positiveStrategy=positiveStrategy,presetBeam=None, useTrigger=True)

        # print(overall_sample_features)
        tmp = []
        max_beam = -1
        for l in txts_at_timestep:
            max_beam = max(max_beam, len(l))
            for story in l:
                tmp.append(story)
        txts_at_timestep = tmp
        chosen = txts_at_timestep
        decision_path.append(len(txts_at_timestep))
        new_chosen = []
        for tup in chosen:
            punct_num, clean_txt = check_puncts(tup[0])
            tup = list(tup)
            tup.append(crt_length)
            if crt_length>minimum_len and punct_num>=initial_num+1:
                # filter out duplicate generations that only differ in punctuations
                if clean_txt not in collection:
                    finished.append(tup)
                    collection.append(clean_txt)
                tup[1]+=1
            else:
                new_chosen.append(tup)
        chosen = new_chosen
        chosen.sort(key=lambda tup: tup[1], reverse=True)
        cache_size = 30
        chosen = chosen[:cache_size]
        prompts_ppls = [(i[0], i[-1]) for i in chosen]
        if crt_length>10:
            break

    prompts_ppls = [[i[0], i[1]/length_penalty(i[-1])-i[-2]*0.01] for i in finished]

    txts = [i[0] for i in prompts_ppls]

    candidates = []
    for txt in txts:
        candidates.append(txt[len(prompt):])

    cloze_distribution = get_cloze_distribution(prompt, candidates)[0]
    new = []
    prompt_valence = get_corpus_valence_score(prompt)
    candidates_valence = [abs(get_corpus_valence_score(i)-prompt_valence) for i in candidates]

    for num, tup in enumerate(prompts_ppls):
        try:
            score = cloze_distribution[num]/30+candidates_valence[num]*3
        except:
            embed()
        new.append((prompts_ppls[num][0],score))
    prompts_ppls = new
    prompts_ppls.sort(key=lambda tup: tup[1], reverse=True)
    txts = [i[0] for i in prompts_ppls]
    words = txts[0].split(" ")
    a = get_arousal_score(words)[0]

    return txts,a, decision_path

def get_ascore(txt):
    words = txt.split(" ")
    tmp = get_arousal_score(words)
    a = tmp[0]/len(words)
    return a

def get_one_sentence(text, selected_pos):
    new = ""
    for i in text:
        if i not in ["?","!"]:
            new+=i
        else:
            new+="."
    text = new
    sentences = text.split(".")
    return sentences[selected_pos]+"."


def get_gpt3_complete_story(context):

  response = openai.Completion.create(
    model="text-davinci-003",
    prompt= "",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  my_openai_obj = list(response.choices)[0]
  story = my_openai_obj.to_dict()['text'].lstrip("\n\n")
  return story

def storygen_switch(context, psositiveStrategy, surprise_pos, story_seg):
    """
    Here the generator is based on GPT2, which sometimes fall short in understanding the story context with a twist.
    To get better stories, please use GPT3 or ChatGPT to finish the story.
    """
    story_segment = story_seg
    txts, a, decision_path = test_mode(story_segment, 10, positiveStrategy=positiveStrategy)

    surprise_st = txts[0]

    gens = finish_story(surprise_st, dg.model, dg.tokenizer, beam_size=30, num_story_return=10, one_st_at_a_time=False)
    tmp = ""
    surprise_st = gens
    filtered_symbols = ["?","<",":",">"]
    for i in surprise_st:
        if i not in filtered_symbols:
            tmp+=i
    surprise_st = tmp
    return surprise_st, story_segment, surprise_pos, a, decision_path, txts
def get_turning_point(k):
    numbers = [1,2,3,4]
    weights = [0.33439115524389623,
 0.35204995649280857,
 0.19977478630291245,
 0.1137841019603829] # the surprise sentence distribution

    sampled_numbers = random.choices(numbers, weights, k=k)
    return sampled_numbers

def extract_sentences(corpus,sts_num):
    sts = sent_tokenize(corpus)
    return " ".join(sts[:sts_num])

if __name__ == "__main__":

    crt_lenght = 1
    b_factors = [10,30,60]

    puncts = [".","!","?"]
    modes = ["planned"]
    dumpee = []
    total_cnt = 0

    cnt=0
    final_rt = []

    contexts = pickle.load(open(Config.contexts_path, "rb"))

    if type(contexts)==tuple:
        contexts = contexts[0]

    # embed()
    story_segs = []

    cnt = 0

    turning_points = get_turning_point(1000)

    for num in range(len(contexts)):

        story_segs.append(contexts[num])
        cnt += 1

    for num, context in tqdm(enumerate(contexts)):

        #get story prompt
        gens = finish_story(context, dg.model, dg.tokenizer, beam_size=30, num_story_return=10,
                            one_st_at_a_time=False)
        whole_prompt = extract_sentences(gens, turning_points[num])

        #get interesting story
        try:
            story_items = storygen_switch(num, positiveStrategy, check_puncts(whole_prompt)[0],whole_prompt)
            print(story_items[0])
            print(story_items[1])
            print(story_items[2])
        except:
            story_items = [0,0,0]

        print("the turning point is: ",turning_points[num])

        dumpee.append((cnt,story_items))
    final_rt.append(dumpee)
    # final_rt.append(surprise_poses)
    import pickle
    pickle.dump(final_rt, open(Config.output_path, "wb"))
