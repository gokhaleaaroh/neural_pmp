import numpy as np
from train import train
from test import benchmarks
from test import visualize

# Mountain Car
# train('mountain_car', num_examples=640, mode=1, num_examples_phase2=1, 
#                retrain_phase1=True, retrain_phase2=True)
# # 
benchmarks(env_name='mountain_car', num_trajs=1, time_steps=list(np.arange(0, 1.0, 0.05)))

# visualize(env_name='mountain_car', time_steps=list(np.arange(0, 1.0, 0.05)), test_trained=False, phase2=False)
visualize(env_name='mountain_car', time_steps=list(np.arange(0, 1.0, 0.05)), test_trained=True, phase2=False)
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
