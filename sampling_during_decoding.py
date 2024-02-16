import os, sys

from utils import top_k_top_p_filtering, get_transformer
import pickle
from config import Config
import torch
from torch.nn import functional as F, CrossEntropyLoss
import string
from transformers import RepetitionPenaltyLogitsProcessor
from nltk import tokenize
# model, tokenizer = get_transformer(Config.GPT2_finetuned_ROC)
from IPython import embed
import numpy as np
loss_fct = CrossEntropyLoss(reduction="none")
PENALTY = 2.0
punctuations = list(string.punctuation) + ['0','1','2','3','4','5','6','7','8','9'] + ['B','C',]

def wrapup_input(input,mode):
    if mode == "original":
        print('aba')
    elif mode == "prompt":
        prompt = '[title]' + input + '[story] '
    return prompt

def count_punct(st):
    cnt = 0
    for i in st:
        if i in punctuations:
            cnt +=1
    return cnt

def punish_logits(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    # embed()
    score = torch.gather(scores, 1, input_ids)

    # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch.where(score < 0, score * PENALTY, score / PENALTY)

    scores.scatter_(1, input_ids, score)
    return scores

def get_storylines(prompt_with_perplexity, crt_length, model,  tokenizer, branch_factor,device):
    '''
        score'(y_t, W_t | y<t) = score(y_t|y<t) + lambda * max(0, max cos_sim(y_t,W))
    '''
    # print(prompt_with_perplexity)

    prompt = prompt_with_perplexity[0]
    # rt = LM.check_real_tok_probabilities(prompt)
    # print(rt)
    # embed()
    perplexity = torch.tensor(prompt_with_perplexity[1]).to(device)
    # print(prompt_with_perplexity)
    # print(prompt)
    # print(perplexity)
    # wrappedup_model_input = wrapup_input(prompt,'prompt')
    max_len = 50
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # embed()
    # model.cuda()
    # model.eval()

    input_ids = input_ids.to(device)
    cur_len = len(input_ids)
    input_tokens = [input_ids]
    valid = True
    # if count_punct(prompt)>1:
    #     return [prompt,perplexity]
    # if crt_buget<0:
    #     beam_output = model.generate(input_ids=input_ids, max_length=50, do_sample=True,
    #                                     num_beams=5, pad_token_id=tokenizer.eos_token_id)
    #     txts = tokenizer.decode(beam_output [0], skip_special_tokens=True)
    #
    # else:

    decoded_tokens_perplexities = expand_storylines(model, input_tokens, branch_factor, \
                                                    perplexity,crt_length,device)

    txts_ppls = []
    for i in range(len(decoded_tokens_perplexities)):
        # embed()
        aba = tokenizer.decode(decoded_tokens_perplexities[i][0][0], skip_special_tokens=False)
        # embed()
        perplex = decoded_tokens_perplexities[i][1]
        txts_ppls.append((aba, perplex))

    return txts_ppls



def expand_storylines(model, input_tokens_list, topk, perplexity,crt_length,device):
    new_token_list = []
    check_list = []

    for input_tokens in input_tokens_list:
        input_ids = input_tokens
        output = model(input_ids)


        # all_probs = torch.softmax(all_logits, dim=1)
        all_logits = output.logits[0].detach()
        all_probs = torch.softmax(all_logits, dim=1)

        logits = output.logits[:, -1, :]

        scores = logits.detach().cpu()
        score = torch.gather(scores, 1, input_ids.detach().cpu())

    # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * PENALTY, score / PENALTY)
        input_ids_cpu = input_ids.detach().cpu()
        logits = scores.scatter_(1, input_ids_cpu, score).to(device)
        del scores
        logits = top_k_top_p_filtering(logits.squeeze(), top_k=100, top_p=0.9)  ###

        logits = F.softmax(logits, dim=-1)


        # y = input_ids[2:]
        # real_topk_probs = logits[np.arange(
        #     0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        # real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))
        #
        next_tokens_ids = torch.topk(logits, topk).indices
        next_tokens_logits = torch.topk(logits, topk).values

        tmp_token_list = []

        for num, next_token_id in enumerate(next_tokens_ids):

            new_seq = torch.concat([input_ids, \
                                    torch.unsqueeze(torch.unsqueeze(next_token_id, dim=0), dim=0)],dim = -1)
            y = new_seq[0][1:].cpu()

            # bos = (new_seq[0] == 50257).nonzero(as_tuple=True)[0].cpu().numpy()[0]
            # sep = (new_seq[0] == 50259).nonzero(as_tuple=True)[0].cpu().numpy()[0]
            real_topk_probs = all_probs[np.arange(
                0, y.shape[0], 1), y].data.cpu().numpy().tolist()
            real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))


            # real_topk_probs = real_topk_probs[0:bos]+real_topk_probs[bos+1:sep] + real_topk_probs[sep+1:]

            real_topk_probs = [i if i >0.00001 else 0.00001 for i in real_topk_probs ][-crt_length:]


            new_perplexity = np.prod(real_topk_probs)**(-1/crt_length)


            # crt_loss = loss_fct(next_tokens_logits,torch.tensor(num).to(device))
            #
            #
            # prev_loss = torch.log(perplexity)*(crt_length-1)
            # new_perplexity = torch.exp((prev_loss + crt_loss)/crt_length)

            tmp_token_list.append((new_seq,\
                                   new_perplexity))

        new_token_list += tmp_token_list
    return new_token_list

