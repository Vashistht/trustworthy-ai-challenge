import matplotlib.pyplot as plt
import pickle 
import numpy
import pandas as pd

file = pd.read_pickle('/home/ubuntu/challenge3/C3/carla/SafeBench/log/exp/exp_sac_ordinary_seed_0/training_results/results.pkl')
episode = file['episode']
episode_reward = file['episode_reward']

plt.plot(episode, episode_reward)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.grid(True)
plt.title('Episode vs Reward')
plt.savefig('/home/ubuntu/challenge3/C3/carla/SafeBench/log/exp/exp_sac_ordinary_seed_0/training_results/reward.png', dpi = 300)