# Improving NeuralPMP

A visualization of the trajectories of the Cart Pole and Mountain Car systems in phase space:
<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/cb53d860-b7c8-48ee-8ea5-6e91c34d84bb" />

<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/01ea6ac8-ab2f-4509-95ac-7b2f1f5d0499" />

A visualizaiton of optimizer updates to the subnetworks in the 5 part compositional network used in NeuralPMP. We visualize Phase 1 and Phase 2 separately for each physical system tested.

Mountain Car:
Phase 1:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/6901e5a2-1c5a-4f85-ae48-5c6ef7ec13e0" />

Phase 2:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/8f7a3dc4-8db4-4f67-ab47-250643be8259" />

Cart Pole:
Phase 1:
<img width="1920" height="1006" alt="image" src="https://github.com/user-attachments/assets/19bfc7e9-9d0e-45e1-bee6-d8a87eb7aaef" />

Phase 2:
<img width="1920" height="1006" alt="image" src="https://github.com/user-attachments/assets/da88fa6d-adc4-4e08-ab6c-df01ada63ae1" />

Pendulum:
Phase 1:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/09cdb1e3-7a54-48be-ae59-6b5bf33221ec" />

Phase 2:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0ab12d90-c7da-47e7-9028-c6dac4c7d937" />

These visualizations can help identify how the learning process progresses within the subnetworks of our larger network. This can allow us to tweak the optimization process to converge more quickly.

This repository is an attempt to build upon https://github.com/mpnguyen2/neural_pmp by converting the learned Hamiltonian dynamics and latent space into a practical controller capable of steering robotic systems with optimal control in a closed-loop manner.

Using the approximation `u_hat = -f_u(q).T * p` does not seem to work, as shown by the following graph:

<img width="1918" height="978" alt="image" src="https://github.com/user-attachments/assets/ef9b6496-2f77-4803-9b22-e4e74c726d82" />

One of the methods that will be attempted is a policy network whose training is guided by the learned Hamiltonian dynamics and latent space. 
