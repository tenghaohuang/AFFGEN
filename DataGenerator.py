import math
import random
import scipy
import numpy as np
from event_trigger_predict import get_event_triggering_score
from sampling_during_decoding import get_storylines
from utils import get_transformer
from config import Config
from valence_measure import get_valence_score, get_arousal_score
from nltk.tokenize import word_tokenize
from ppl_sentence_level import get_story_ppl, process_story
import string
import torch
from IPython import embed
# from GLTR import LM
class DataGenerator():
    """
    Generate badit data.

    Defaults:
    K=2 arms
    D=2 features/arm
    """
    def __init__(self,K=2,D=2,feature_type='binary',reward_type='binary'):
        
        self.D = D # dimension of the feature vector
        self.K = K # number of bandits
        self.reward_type = reward_type
        self.feature_type = feature_type
        self.means = np.random.normal(size=self.K)
        self.stds = 1 + 2*np.random.rand(self.K)
        self.device =  torch.device("cuda")

        # generate the weight vectors.  initialize estimate of feature
        # importance for each arm's d features
        self.generate_weight_vectors()
        self.model, self.tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()
        self.topk = 10
        self.branch_factors = [10, 30, 60]
        self.evaluating = False
        # model, tokenizer = get_transformer()
        # model.to('cuda')
        # self.JUDGE = LM(model, tokenizer)
    def generate_weight_vectors(self,loc=0.0,scale=1.0):
        self.W = np.random.normal(loc=loc,scale=scale,size=(self.K,self.D))
        #self.W = np.ones((self.K,self.D))

    def group_bucket(self, scores, anchors, b):
        l = []
        start = 0
        for num in range(len(anchors)):
            aba = scores[start:start + min(b, anchors[num])]
            if type(aba) is not list:
                aba = [aba]
            l += aba
            start += anchors[num]
        return l

    def get_topk_reward_each_bucket(self, buckets, budget, crt_length, previous_pivot):
        sample_rewards = []

        # branch_facts = [True if budget > b else False for b in self.branch_factors]

        # print(tup)
        # embed()#look at perplex
        for num in range(len(buckets)):
            b = self.branch_factors[num]
            buckets[num].sort(key=lambda tup: tup[1], reverse=True)
            reward = np.exp(buckets[num][0][1]-0.0003*b-previous_pivot)
            sample_rewards.append(reward)
        sample_rewards = [5*i/sum(sample_rewards) for i in sample_rewards]

        return sample_rewards

    def put_buckets(self, buckets, txts_at_timestep):
        for num, b in enumerate(self.branch_factors):
            buckets[num]+=txts_at_timestep[:b]
        return buckets
    def generate_features_rewards(self,text_prompts_ppls, crt_length, crt_budget, \
                                  mode="train",positiveStrategy=None,presetBeam=None, useTrigger=False):
        txts_at_timestep = []
        overall_features = []
        # print(text_prompts_ppls)
        print(crt_length)
        overall_rewards = []
        # embed()
        for txt_prompt_ppl in text_prompts_ppls:
            # Let us have beam_size * features and beam_size * ground_truth_decision
            # beam filter would happen after each timestep
            buckets = [[], [], []]
            txt = txt_prompt_ppl[0]
            if useTrigger:
                trigger_score = get_event_triggering_score(txt)
            else:
                trigger_score = 0
            txt_a = get_arousal_score(txt.split(" "))[0]
            txt_v = get_valence_score(txt.split(" "))[0]

            rt_txts_ppls = get_storylines(txt_prompt_ppl, crt_length, self.model, self.tokenizer, self.branch_factors[-1], device=torch.device("cuda"))
            crt_appendee = []

            for tmp in rt_txts_ppls:
                appendee_txt = tmp[0]
                appendee_txt_ppl = tmp[1]
                # v = get_valence_score(tmp.split(" "))[0]
                a = get_arousal_score(appendee_txt.split(" "))[0]

                gem = (appendee_txt, a - 0.00015 * appendee_txt_ppl, txt_v, txt_a, trigger_score, appendee_txt_ppl)
                # txts_at_timestep.append(gem)
                crt_appendee.append(gem)
            buckets = self.put_buckets(buckets, crt_appendee)
            overall_features.append((crt_length/10, trigger_score, txt_v, txt_a, txt_prompt_ppl[1]))
            prompt_sample_rewards = self.get_topk_reward_each_bucket(buckets, crt_budget, crt_length,
                                                                                      previous_pivot=txt_a)
            if mode == "train":
                if trigger_score > 0.0001:
                    overall_rewards.append(prompt_sample_rewards)
                    if self.evaluating == False:
                        txts_at_timestep.append(buckets[np.argmax(prompt_sample_rewards)])
                else:
                    txts_at_timestep.append(buckets[0])
                    overall_rewards.append([max(prompt_sample_rewards)-0.0003*10,max(prompt_sample_rewards)-0.0003*30, max(prompt_sample_rewards)-0.0003*60])
            elif mode == 'test':
                # if trigger_score > 0.0001:
                feature = [(crt_length/10, trigger_score, txt_v, txt_a, txt_prompt_ppl[1])]
                l = [positiveStrategy.estimate(0, feature),
                     positiveStrategy.estimate(1, feature), \
                     positiveStrategy.estimate(2, feature)]
                arm = np.argmax(l)
                txts_at_timestep.append(buckets[arm])
                # else:
                #     txts_at_timestep.append(buckets[0])
            elif mode =='baseline':

                if useTrigger:
                    if trigger_score > 0.0001:
                        txts_at_timestep.append(buckets[presetBeam])
                    else:
                        txts_at_timestep.append(buckets[0])
                else:
                    txts_at_timestep.append(buckets[presetBeam])

        return txts_at_timestep, overall_features, np.asarray(overall_rewards), buckets

    def filter_by_perplexity(self, candidate_list,crt_length):
        rt_l = []
        for story in candidate_list:
            context_sentences = process_story(story)
            ppl_scores = get_story_ppl(context_sentences, self.tokenizer, self.model, self.device)
            if ppl_scores[-1]<(crt_length)*1000:
                rt_l.append(story)
        if len(rt_l)<int(len(candidate_list)/3):
            rt_l = []
        return rt_l


