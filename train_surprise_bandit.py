import math
import random

import DataGenerator
import PositiveStrategy
import Simulator
from utils import getStories
from config import Config
from IPython import embed
import numpy as np
import pickle
from tqdm import tqdm

arms = 3
features = 5
rewardType = 'positive'
featureType = 'integer'
# Get training data
contexts, references = getStories(Config.story_path, story_num=50000, surprise_position=1)
dg = DataGenerator.DataGenerator(arms, features, feature_type=featureType, reward_type=rewardType)
positiveStrategy = PositiveStrategy.PositiveStrategy(arms, features)
simulator = Simulator.Simulator(positiveStrategy)
cache_size = 30

for epoch in range(10):

    records = []
    slice = 3000
    for id in tqdm(range(0, slice)):
        print("experiment: %d" % id)
        previous_rmse = 0.
        crt_budget = 0
        crt_length = 0
        crt_id = int(epoch * slice + id)
        print(contexts[crt_id])
        if contexts[crt_id].endswith(" "):
            prompts_ppls = [[contexts[crt_id][:-1], 1]]
        else:
            prompts_ppls = [[contexts[crt_id], 1]]
        trace = []
        while True:
            crt_length += 1

            txts_at_timestep, overall_features, overall_rewards, buckets = dg.generate_features_rewards(prompts_ppls,
                                                                                                        crt_length,
                                                                                                        crt_budget)
            if crt_length > 10:
                break

            regret, rmse, armChoices = simulator.simulate(overall_features, overall_rewards, dg.W)
            tmp = []
            sample_factor = 0.33

            print("the length of generated prompts are", len(txts_at_timestep))
            avg_beam_keep = 0
            txts_at_timestep.sort(key=lambda tup: tup[1], reverse=True)
            if dg.evaluating == False:
                # embed()
                for num, r in enumerate(overall_rewards):
                    avg_beam_keep += dg.branch_factors[np.argmax(r)]
                avg_beam_keep = avg_beam_keep / len(overall_rewards)
                if type(txts_at_timestep[0][0] is list):
                    txts_at_timestep = txts_at_timestep[0]

                txts_at_timestep = txts_at_timestep[:cache_size]

            print("the length of filtered prompts are", len(txts_at_timestep))
            prompts_ppls = [(i[0], i[-1]) for i in txts_at_timestep]

            if previous_rmse == 0:
                initial_rmse = rmse[0][-1]
                previous_rmse = rmse[0][-1]

            print(txts_at_timestep[:3])
