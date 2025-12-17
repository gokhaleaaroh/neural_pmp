import numpy as np
import utils
from train import train
from test import benchmarks
from test import visualize
import torch

# Mountain Car
# train('mountain_car', num_examples=640, mode=1, num_examples_phase2=1, 
#                retrain_phase1=True, retrain_phase2=True)
# # 


env = utils.get_environment('mountain_car') 

q = torch.tensor(env.sample_q(1, mode='test'), dtype=torch.float)

# print("Initial position: ", q)
# print("Trying to move to the right...")
# 
# for i in range(0, 100):
#     q = env.step(q, torch.tensor([[0.8]]), dt=0.05)
#     print("Updated position: ", q)

benchmarks(env_name='mountain_car', num_trajs=2000, time_steps=list(np.arange(0, 1.0, 0.05)))

# visualize(env_name='mountain_car', time_steps=list(np.arange(0, 1.0, 0.05)), test_trained=False, phase2=False)
# visualize(env_name='mountain_car', time_steps=list(np.arange(0, 1.0, 0.05)), test_trained=True, phase2=False)
# visualize(env_name='mountain_car', time_steps=list(np.arange(0, 1.0, 0.05)), test_trained=True, phase2=True)
# 
# 
 # Cart Pole
# train('cartpole', num_examples=640, mode=1, num_examples_phase2=1,
#               retrain_phase1=True, retrain_phase2=True)
 
# benchmarks(env_name='cartpole', num_trajs=2000, time_steps=list(np.arange(0, 10.0, 0.05)))
#  
# visualize(env_name='cartpole', time_steps=list(np.arange(0, 10.0, 0.05)), test_trained=False, phase2=False)
# visualize(env_name='cartpole', time_steps=list(np.arange(0, 10.0, 0.05)), test_trained=True, phase2=False)
# visualize(env_name='cartpole', time_steps=list(np.arange(0, 10.0, 0.05)), test_trained=True, phase2=True)

# Pendulum
# train('pendulum', num_examples=640, mode=1, num_examples_phase2=1,
#               retrain_phase1=True, retrain_phase2=True)
# 
# benchmarks(env_name='pendulum', num_trajs=2000, time_steps=list(np.arange(0, 1.0, 0.05)))

# visualize(env_name='cartpole', time_steps=list(np.arange(0, 10.0, 0.05)), test_trained=False, phase2=False)
# visualize(env_name='cartpole', time_steps=list(np.arange(0, 10.0, 0.05)), test_trained=True, phase2=False)
# visualize(env_name='cartpole', time_steps=list(np.arange(0, 10.0, 0.05)), test_trained=True, phase2=True)
