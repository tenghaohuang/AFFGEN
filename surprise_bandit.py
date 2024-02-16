
import DataGenerator
import PositiveStrategy
import Simulator
from utils import getStories
from config import Config
import numpy as np
from tqdm import tqdm
arms=3
features=5
rewardType='positive'
featureType='integer'
cache_size = 30


# define number of samples and number of choices
contexts, references = getStories(Config.story_path, story_num=50000, surprise_position=1)

dg = DataGenerator.DataGenerator(arms, features, feature_type=featureType, reward_type=rewardType)
positiveStrategy = PositiveStrategy.PositiveStrategy(arms, features)
# print(positiveStrategy)
simulator = Simulator.Simulator(positiveStrategy)

for epoch in range(10):

    num_batches = 100
    num_experiments = 1

    total_regret = []
    total_rmse = []

    records = []
    slice = 3000
    for id in tqdm(range(0, slice)):
        print("experiment: %d" % id)
        previous_rmse = 0.

        crt_budget = 0 # removed budget from pipeline design
        crt_length = 0
        crt_id = int(epoch*slice + id)
        print(contexts[crt_id])
        if contexts[crt_id].endswith(" "):
            prompts_ppls = [[contexts[crt_id][:-1],1]]
        else:
            prompts_ppls = [[contexts[crt_id],1]]
        trace = []
        while True:
            crt_length += 1

            txts_at_timestep, overall_features, overall_rewards, buckets = dg.generate_features_rewards(prompts_ppls, crt_length, crt_budget)
            if crt_length>10:
                break

            regret, rmse, armChoices = simulator.simulate(overall_features, overall_rewards, dg.W)
            tmp = []
            sample_factor = 0.33

            print("the length of generated prompts are", len(txts_at_timestep))
            avg_beam_keep = 0
            txts_at_timestep.sort(key=lambda tup: tup[1], reverse=True)
            if dg.evaluating == False:

                for num, r in enumerate(overall_rewards):
                    avg_beam_keep += dg.branch_factors[np.argmax(r)]
                avg_beam_keep = avg_beam_keep/len(overall_rewards)

                if type(txts_at_timestep[0][0] is list):
                    txts_at_timestep = txts_at_timestep[0]
                txts_at_timestep = txts_at_timestep[:cache_size]

            print("the length of filtered prompts are", len(txts_at_timestep))
            prompts_ppls = [(i[0],i[-1]) for i in txts_at_timestep]

            if previous_rmse == 0:
                initial_rmse = rmse[0][-1]
                previous_rmse = rmse[0][-1]

            if (len(total_rmse) == 0):
                total_rmse = [rmse]
                total_regret = [regret]
            else:
                total_rmse.append(np.mean(rmse))
                total_regret.append(np.mean(regret))
                # print(len(total_regret))
            print(txts_at_timestep[:3])


