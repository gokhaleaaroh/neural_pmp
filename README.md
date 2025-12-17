# Crafting a Controller with NeuralPMP

This repository is an attempt to build upon https://github.com/mpnguyen2/neural_pmp by converting the learned Hamiltonian dynamics and latent space into a practical controller capable of steering robotic systems with optimal control in a closed-loop manner.

Using the approximation `u_hat = -f_u(q).T * p` does not seem to work, as shown by the following graph:

<img width="1918" height="978" alt="image" src="https://github.com/user-attachments/assets/ef9b6496-2f77-4803-9b22-e4e74c726d82" />

One of the methods that will be attempted is a policy network whose training is guided by the learned Hamiltonian dynamics and latent space. 
